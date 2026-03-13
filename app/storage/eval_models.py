"""
Evaluation system DB models.

EvalCase  — a ground-truth test case (question + expected answer + video)
EvalRun   — one execution of a case, with all scored metrics stored as JSONB
"""

import uuid
from datetime import datetime
from sqlalchemy import Column, String, Float, DateTime, Text, Integer, Boolean
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import declarative_base
from app.storage.models import Base


class EvalCase(Base):
    """A ground-truth test case to evaluate the QA system against."""
    __tablename__ = "eval_cases"

    id          = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    title       = Column(String(200), nullable=False)          # short human label
    question    = Column(Text, nullable=False)
    expected    = Column(Text, nullable=False)                  # expected answer / key facts
    video_filename = Column(String, nullable=True)             # None = all videos
    tags        = Column(JSONB, nullable=True)                  # ["latency","hallucination",…]
    created_at  = Column(DateTime, default=datetime.utcnow)
    updated_at  = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class EvalRun(Base):
    """One scored evaluation run for a single EvalCase."""
    __tablename__ = "eval_runs"

    id              = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    case_id         = Column(UUID(as_uuid=True), nullable=False, index=True)

    # Raw outputs
    actual_answer   = Column(Text, nullable=True)
    context_sent    = Column(Text, nullable=True)       # full context string sent to LLM
    sources         = Column(JSONB, nullable=True)      # source tracks returned
    fast_path       = Column(Boolean, default=False)

    # Timing
    latency_ms      = Column(Float, nullable=True)      # total wall time
    llm_ms          = Column(Float, nullable=True)      # LLM-only time
    error           = Column(String(200), nullable=True)

    # Scores  (all 0.0–1.0, -1 = not scored yet)
    score_accuracy      = Column(Float, default=-1)   # LLM judge: does answer match expected?
    score_groundedness  = Column(Float, default=-1)   # claims in answer backed by context
    score_hallucination = Column(Float, default=-1)   # 1 - hallucination rate (higher=better)
    score_latency       = Column(Float, default=-1)   # normalised: 1=fast, 0=slow/timeout

    # Per-metric LLM judge rationale
    score_precision    = Column(Float, default=-1)   # retrieval precision
    score_recall       = Column(Float, default=-1)   # answer recall vs expected
    judge_notes     = Column(JSONB, nullable=True)     # {"accuracy":"…","groundedness":"…"}

    run_at          = Column(DateTime, default=datetime.utcnow)
    model_name      = Column(String(100), nullable=True)