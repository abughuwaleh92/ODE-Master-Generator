# src/database/models.py
from sqlalchemy import Column, Integer, String, Float, DateTime, JSON
from sqlalchemy.ext.declarative import declarative_base
from datetime import datetime

Base = declarative_base()

class GeneratedODE(Base):
    __tablename__ = 'generated_odes'
    
    id = Column(Integer, primary_key=True)
    ode_expression = Column(String)
    solution = Column(String)
    type = Column(String)
    order = Column(Integer)
    parameters = Column(JSON)
    novelty_score = Column(Float)
    created_at = Column(DateTime, default=datetime.utcnow)
