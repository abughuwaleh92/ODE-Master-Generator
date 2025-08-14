
# -*- coding: utf-8 -*-
"""
api_server.py
=============

FastAPI backend for Master Generators.
- Build ODE via Theorem 4.2 from a free-form template and f(z).
- Persist and list ODEs (SQLite / Postgres via DATABASE_URL).
- ML: train/predict enriched labels.
- DL: train/score novelty.
- Triage: compute complexity & novelty for stored ODEs.

Run:
    uvicorn api_server:app --reload --host 0.0.0.0 --port 8000
"""

import os, json, pickle, datetime as dt
from typing import Any, Dict, List, Optional, Tuple

import sympy as sp
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

from core_master_generators import (
    Theorem42, GeneratorBuilder, safe_eval_f_of_z,
    ode_to_json, expr_from_srepr, count_symbolic_complexity,
    GeneratorPatternLearner, NoveltyDetector
)

# ----------------------------------------------------------------------------
# Database (SQLAlchemy)
# ----------------------------------------------------------------------------
from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime
from sqlalchemy.orm import declarative_base, sessionmaker

DATABASE_URL = os.environ.get("DATABASE_URL", "sqlite:////mnt/data/masters.db")
engine = create_engine(DATABASE_URL, future=True)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)
Base = declarative_base()

class ODERecord(Base):
    __tablename__ = "odes"
    id = Column(Integer, primary_key=True, index=True)
    lhs_srepr = Column(Text, nullable=False)
    rhs_srepr = Column(Text, nullable=False)
    lhs_latex = Column(Text, nullable=False)
    rhs_latex = Column(Text, nullable=False)
    meta_json = Column(Text, nullable=True)
    created_at = Column(DateTime, default=dt.datetime.utcnow)

Base.metadata.create_all(bind=engine)

# ----------------------------------------------------------------------------
# Models
# ----------------------------------------------------------------------------
class BuildRequest(BaseModel):
    f_str: str = Field(..., description="f(z) as a SymPy expression string")
    template: str = Field(..., description="Generator template using y, Dy1, Dy2, ..., Dym")
    n: str = Field("n", description="Integer like '4' or 'n' for symbolic")
    m: Optional[str] = Field(None, description="Integer like '2' or None for symbolic m")
    alpha: str = Field("alpha", description="alpha expression")
    beta: str = Field("beta", description="beta expression")
    complex_form: bool = True
    persist: bool = False

class ODERecordIn(BaseModel):
    lhs_srepr: str
    rhs_srepr: str
    lhs_latex: str
    rhs_latex: str
    meta: Optional[Dict[str, Any]] = None

class TrainMLItem(BaseModel):
    lhs_srepr: str
    rhs_srepr: str
    labels: Dict[str, int]

class TrainMLRequest(BaseModel):
    items: List[TrainMLItem]

class PredictItem(BaseModel):
    lhs_srepr: str
    rhs_srepr: str

class PredictRequest(BaseModel):
    items: List[PredictItem]

class TrainDLItem(BaseModel):
    ode_str: str
    target: float

class TrainDLRequest(BaseModel):
    items: List[TrainDLItem]

class ScoreDLRequest(BaseModel):
    ode_strs: List[str]

# ----------------------------------------------------------------------------
# App state
# ----------------------------------------------------------------------------
app = FastAPI(title="Master Generators API")
ML_PATH = "/mnt/data/models/ml.pkl"
DL_PATH = "/mnt/data/models/dl.pt"
os.makedirs("/mnt/data/models", exist_ok=True)

class State:
    def __init__(self):
        self.ml = GeneratorPatternLearner()
        self.dl = NoveltyDetector()

state = State()

# Try restore ML
if os.path.exists(ML_PATH):
    try:
        with open(ML_PATH, 'rb') as f:
            state.ml = pickle.load(f)
    except Exception:
        pass
# Try restore DL (PyTorch)
try:
    import torch
    if os.path.exists(DL_PATH) and state.dl.available:
        state.dl.model.load_state_dict(torch.load(DL_PATH, map_location='cpu'))
except Exception:
    pass

# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------
def parse_nm(n_str: str, m_str: Optional[str], T: Theorem42):
    n_val = T.n if n_str.strip().lower() == 'n' else int(n_str)
    m_val = None if (m_str is None or not str(m_str).strip()) else int(m_str)
    return n_val, m_val

def parse_ab(a_str: str, b_str: str, T: Theorem42):
    ns = {'alpha': T.alpha, 'beta': T.beta, 'pi': sp.pi, 'I': sp.I}
    a = sp.sympify(a_str, locals=ns)
    b = sp.sympify(b_str, locals=ns)
    return a, b

# ----------------------------------------------------------------------------
# Endpoints
# ----------------------------------------------------------------------------
@app.get("/health")
def health():
    return {"status": "ok", "now": dt.datetime.utcnow().isoformat()}

@app.post("/build_ode")
def build_ode(req: BuildRequest):
    try:
        baseT = Theorem42()
        n_val, m_val = parse_nm(req.n, req.m, baseT)
        alpha, beta = parse_ab(req.alpha, req.beta, baseT)
        T = Theorem42(alpha=alpha, beta=beta, n=n_val)
        f = safe_eval_f_of_z(req.f_str); f_callable = lambda z: f(z)
        builder = GeneratorBuilder(T, f_callable, n_val, m_val, req.complex_form)
        lhs, rhs = builder.build(req.template)
        data = ode_to_json(lhs, rhs, meta={
            "f_str": req.f_str, "template": req.template, "n": req.n, "m": req.m or "symbolic",
            "alpha": str(alpha), "beta": str(beta), "complex_form": req.complex_form
        })
        if req.persist:
            with SessionLocal() as db:
                rec = ODERecord(lhs_srepr=data["lhs_srepr"], rhs_srepr=data["rhs_srepr"],
                                lhs_latex=data["lhs_latex"], rhs_latex=data["rhs_latex"],
                                meta_json=json.dumps(data.get("meta", {})))
                db.add(rec); db.commit(); db.refresh(rec)
                data["id"] = rec.id
        return data
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/store/ode")
def store_ode(rec: ODERecordIn):
    try:
        with SessionLocal() as db:
            row = ODERecord(lhs_srepr=rec.lhs_srepr, rhs_srepr=rec.rhs_srepr,
                            lhs_latex=rec.lhs_latex, rhs_latex=rec.rhs_latex,
                            meta_json=json.dumps(rec.meta or {}))
            db.add(row); db.commit(); db.refresh(row)
            return {"status":"ok", "id": row.id}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/odes")
def list_odes(limit: int = Query(20, ge=1, le=200)):
    with SessionLocal() as db:
        rows = db.query(ODERecord).order_by(ODERecord.id.desc()).limit(limit).all()
        out = []
        for r in rows:
            out.append({"id": r.id, "lhs_latex": r.lhs_latex, "rhs_latex": r.rhs_latex, "meta": json.loads(r.meta_json or "{}")})
        return out

@app.get("/ode/{oid}")
def get_ode(oid: int):
    with SessionLocal() as db:
        r = db.query(ODERecord).filter(ODERecord.id==oid).first()
        if not r:
            raise HTTPException(status_code=404, detail="not found")
        return {"id": r.id, "lhs_srepr": r.lhs_srepr, "rhs_srepr": r.rhs_srepr,
                "lhs_latex": r.lhs_latex, "rhs_latex": r.rhs_latex,
                "meta": json.loads(r.meta_json or "{}"), "created_at": r.created_at.isoformat()}

@app.post("/ml/train")
def ml_train(req: TrainMLRequest):
    try:
        dataset = []
        for it in req.items:
            L = expr_from_srepr(it.lhs_srepr)
            R = expr_from_srepr(it.rhs_srepr)
            dataset.append({"lhs": L, "rhs": R, "labels": it.labels})
        info = state.ml.train(dataset)
        # Persist ML model if sklearn available
        try:
            with open(ML_PATH, 'wb') as f:
                pickle.dump(state.ml, f)
        except Exception:
            pass
        return {"status": "ok", "info": info}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/ml/predict")
def ml_predict(req: PredictRequest):
    try:
        pairs = [(expr_from_srepr(it.lhs_srepr), expr_from_srepr(it.rhs_srepr)) for it in req.items]
        preds = state.ml.predict(pairs)
        return {"preds": preds}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/dl/train")
def dl_train(req: TrainDLRequest):
    try:
        pairs = [(it.ode_str, float(it.target)) for it in req.items]
        info = state.dl.quick_train(pairs, epochs=3)
        # Save torch state if available
        try:
            import torch
            torch.save(state.dl.model.state_dict(), DL_PATH)
        except Exception:
            pass
        return {"status": "ok", "info": info}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/dl/score")
def dl_score(req: ScoreDLRequest):
    try:
        scores = state.dl.novelty_score(req.ode_strs)
        return {"scores": scores}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/triage/{oid}")
def triage(oid: int):
    with SessionLocal() as db:
        r = db.query(ODERecord).filter(ODERecord.id==oid).first()
        if not r:
            raise HTTPException(status_code=404, detail="not found")
        L = expr_from_srepr(r.lhs_srepr); R = expr_from_srepr(r.rhs_srepr)
        feats_L = count_symbolic_complexity(L); feats_R = count_symbolic_complexity(R)
        score = state.dl.novelty_score([sp.srepr(sp.Eq(L, R))])[0]
        return {"features": {"lhs": feats_L, "rhs": feats_R}, "novelty": score}
