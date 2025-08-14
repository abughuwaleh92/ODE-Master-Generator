# api_server.py
# FastAPI backend for Master Generators (generation, training, novelty, storage)
from __future__ import annotations

import os
import json
import sqlite3
import datetime as dt
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import sympy as sp
from sympy import Eq

# Import core
try:
    from core_master_generators import (
        Theorem42, GeneratorBuilder, TemplateConfig,
        safe_eval_f_of_z, ode_to_json, expr_from_srepr,
        GeneratorPatternLearner, NoveltyDetector
    )
except Exception:
    from mg_core.core_master_generators import (
        Theorem42, GeneratorBuilder, TemplateConfig,
        safe_eval_f_of_z, ode_to_json, expr_from_srepr,
        GeneratorPatternLearner, NoveltyDetector
    )

# --------------------- Storage (SQLite) ---------------------

DB_URL = os.getenv("DATABASE_URL", "sqlite:///data/odes.db")
DB_PATH = DB_URL.replace("sqlite:///", "")

os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)

def _db_conn():
    return sqlite3.connect(DB_PATH)

def init_db():
    with _db_conn() as con:
        con.execute("""
        CREATE TABLE IF NOT EXISTS odes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT NOT NULL,
            template TEXT NOT NULL,
            f_str TEXT NOT NULL,
            n_repr TEXT NOT NULL,
            m_repr TEXT NOT NULL,
            complex_form INTEGER NOT NULL,
            lhs_srepr TEXT NOT NULL,
            rhs_srepr TEXT NOT NULL,
            lhs_latex TEXT NOT NULL,
            rhs_latex TEXT NOT NULL,
            meta_json TEXT NOT NULL
        )
        """)
init_db()

def save_ode_row(payload: Dict[str, Any]) -> int:
    with _db_conn() as con:
        cur = con.cursor()
        cur.execute("""
            INSERT INTO odes (created_at, template, f_str, n_repr, m_repr, complex_form,
                              lhs_srepr, rhs_srepr, lhs_latex, rhs_latex, meta_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            dt.datetime.utcnow().isoformat() + "Z",
            payload["template"],
            payload["f_str"],
            payload["n_repr"],
            payload["m_repr"],
            1 if payload["complex_form"] else 0,
            payload["lhs_srepr"],
            payload["rhs_srepr"],
            payload["lhs_latex"],
            payload["rhs_latex"],
            json.dumps(payload.get("meta", {}), ensure_ascii=False)
        ))
        return cur.lastrowid

def list_odes(limit: int = 20, offset: int = 0) -> List[Dict[str, Any]]:
    with _db_conn() as con:
        cur = con.cursor()
        cur.execute("""
            SELECT id, created_at, template, f_str, n_repr, m_repr, complex_form,
                   lhs_srepr, rhs_srepr, lhs_latex, rhs_latex, meta_json
            FROM odes
            ORDER BY id DESC
            LIMIT ? OFFSET ?
        """, (limit, offset))
        rows = cur.fetchall()
    out = []
    for r in rows:
        out.append({
            "id": r[0],
            "created_at": r[1],
            "template": r[2],
            "f_str": r[3],
            "n_repr": r[4],
            "m_repr": r[5],
            "complex_form": bool(r[6]),
            "lhs_srepr": r[7],
            "rhs_srepr": r[8],
            "lhs_latex": r[9],
            "rhs_latex": r[10],
            "meta": json.loads(r[11] or "{}"),
        })
    return out

# --------------------- ML/DL Singletons ---------------------

LEARNER = GeneratorPatternLearner()
NOVELTY = NoveltyDetector()

# --------------------- Pydantic Models ---------------------

class GenerateRequest(BaseModel):
    template: str
    f_str: str
    n_mode: str = "symbolic"   # "symbolic" or "numeric"
    n_value: int = 2
    m_mode: str = "symbolic"   # "symbolic" or "numeric"
    m_value: int = 3
    complex_form: bool = True
    persist: bool = False
    meta: Optional[Dict[str, Any]] = None

class GenerateResponse(BaseModel):
    id: Optional[int] = None
    lhs_latex: str
    rhs_latex: str
    lhs_srepr: str
    rhs_srepr: str
    lhs_str: str
    rhs_str: str
    complexity_lhs: Dict[str, int]
    complexity_rhs: Dict[str, int]
    meta: Dict[str, Any]

class TrainPayload(BaseModel):
    samples: List[Dict[str, Any]]  # each: {lhs_srepr, rhs_srepr, meta}

class ClassifyRequest(BaseModel):
    lhs_srepr: str
    rhs_srepr: str

class ClassifyResponse(BaseModel):
    linear: int
    stiffness: int
    solvability: int

class NoveltyRequest(BaseModel):
    lhs_srepr: str
    rhs_srepr: str

class NoveltyResponse(BaseModel):
    score: float

# --------------------- FastAPI App ---------------------

app = FastAPI(title="Master Generators API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"],
)

@app.get("/health")
def health():
    return {"ok": True, "version": app.version}

# --------------------- Endpoints ---------------------

@app.post("/generate", response_model=GenerateResponse)
def generate(req: GenerateRequest):
    try:
        # Construct theorem object
        if req.n_mode == "symbolic":
            T = Theorem42(n="n")
            n_repr = "n"
        else:
            if req.n_value <= 0:
                raise HTTPException(status_code=400, detail="n_value must be positive integer.")
            T = Theorem42(n=int(req.n_value))
            n_repr = str(int(req.n_value))

        f = safe_eval_f_of_z(req.f_str)
        G = GeneratorBuilder(T, TemplateConfig(alpha=T.alpha, beta=T.beta, n=T.n, m_sym=T.m_sym))

        if req.m_mode == "symbolic":
            m_arg = T.m_sym
            m_repr = "m"
        else:
            if req.m_value <= 0:
                raise HTTPException(status_code=400, detail="m_value must be positive integer.")
            m_arg = int(req.m_value)
            m_repr = str(int(req.m_value))

        lhs, rhs = G.build(
            template=req.template,
            f=f,
            m=m_arg,
            n_override=None,
            complex_form=req.complex_form
        )

        payload = json.loads(ode_to_json(lhs, rhs, meta=req.meta or {}))
        # augment with ids + params
        payload["meta"].update({
            "template": req.template, "f_str": req.f_str,
            "n_repr": n_repr, "m_repr": m_repr,
            "complex_form": req.complex_form
        })

        row_id = None
        if req.persist:
            save_payload = {
                "template": req.template,
                "f_str": req.f_str,
                "n_repr": n_repr,
                "m_repr": m_repr,
                "complex_form": req.complex_form,
                "lhs_srepr": payload["lhs_srepr"],
                "rhs_srepr": payload["rhs_srepr"],
                "lhs_latex": payload["lhs_latex"],
                "rhs_latex": payload["rhs_latex"],
                "meta": payload["meta"]
            }
            row_id = save_ode_row(save_payload)

        return GenerateResponse(
            id=row_id,
            lhs_latex=payload["lhs_latex"],
            rhs_latex=payload["rhs_latex"],
            lhs_srepr=payload["lhs_srepr"],
            rhs_srepr=payload["rhs_srepr"],
            lhs_str=payload["lhs_str"],
            rhs_str=payload["rhs_str"],
            complexity_lhs=payload["complexity_lhs"],
            complexity_rhs=payload["complexity_rhs"],
            meta=payload["meta"]
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Generation failed: {e}")

@app.post("/train")
def train(payload: TrainPayload):
    try:
        ds = []
        for item in payload.samples:
            lhs = expr_from_srepr(item["lhs_srepr"])
            rhs = expr_from_srepr(item["rhs_srepr"])
            meta = item.get("meta", {})
            ds.append((lhs, rhs, meta))
        LEARNER.train(ds)
        return {"ok": True, "trained_on": len(ds)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Training failed: {e}")

@app.post("/classify", response_model=ClassifyResponse)
def classify(req: ClassifyRequest):
    try:
        lhs = expr_from_srepr(req.lhs_srepr)
        rhs = expr_from_srepr(req.rhs_srepr)
        pred = LEARNER.predict([(lhs, rhs)])[0]
        return ClassifyResponse(**pred)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Classification failed: {e}")

@app.post("/novelty", response_model=NoveltyResponse)
def novelty(req: NoveltyRequest):
    try:
        lhs = expr_from_srepr(req.lhs_srepr)
        rhs = expr_from_srepr(req.rhs_srepr)
        score = NOVELTY.score(lhs, rhs)
        return NoveltyResponse(score=float(score))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Novelty scoring failed: {e}")

@app.get("/odes")
def get_odes(limit: int = Query(20, ge=1, le=200), offset: int = Query(0, ge=0)):
    try:
        return {"items": list_odes(limit=limit, offset=offset)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Listing failed: {e}")
