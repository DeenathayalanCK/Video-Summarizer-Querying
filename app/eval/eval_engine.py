"""
Evaluation engine — auto-populates from ollama_call logs.

Two evaluation modes:
  AUTO   — reads ask/ask_captions calls from ollama_calls.jsonl.
            Scores: latency, groundedness, hallucination (no ground truth needed).
  MANUAL — user-defined EvalCase with expected answer.
            Adds: accuracy, precision, recall via LLM judge.

Question extraction: parses "Question: <text>" from the logged prompt.
Context extraction:  parses the CONTEXT block from the logged prompt.
"""

import re
import json
import time
import requests
import statistics
from typing import Optional
from pathlib import Path

from app.core.config import get_settings
from app.core.logging import get_logger

_LOG_FILE = Path("/data/logs/ollama_calls.jsonl")

# ── Latency scoring ───────────────────────────────────────────────────────────
_LATENCY_BRACKETS = [(2000,1.0),(5000,0.8),(15000,0.5),(30000,0.2),(60000,0.1)]

def _score_latency(ms: float) -> float:
    for threshold, score in _LATENCY_BRACKETS:
        if ms <= threshold:
            return score
    return 0.0

# ── Extract question from logged prompt ───────────────────────────────────────
_Q_RE = re.compile(r"Question:\s*(.+?)(?:\n|$)", re.IGNORECASE)

def _extract_question(prompt: str) -> str:
    m = _Q_RE.search(prompt)
    return m.group(1).strip() if m else prompt[:120].strip()

# ── Extract context block from logged prompt ──────────────────────────────────
_CTX_RE = re.compile(
    r"--- CONTEXT ---\s*(.*?)\s*--- END CONTEXT ---",
    re.DOTALL | re.IGNORECASE,
)

def _extract_context(prompt: str) -> str:
    m = _CTX_RE.search(prompt)
    return m.group(1).strip() if m else ""

# ── LLM judge ────────────────────────────────────────────────────────────────
_JUDGE_SYSTEM = (
    "You are a strict evaluation judge for a surveillance video QA system. "
    "Score AI-generated answers. Reply ONLY with a valid JSON object, no markdown."
)

_AUTO_JUDGE_PROMPT = """
Evaluate this answer and return JSON with exactly these keys:

{{
  "groundedness": <0.0-1.0>,
  "groundedness_reason": "<one sentence>",
  "hallucination": <0.0-1.0>,
  "hallucination_reason": "<one sentence>",
  "retrieval_precision": <0.0-1.0>,
  "retrieval_precision_reason": "<one sentence>"
}}

groundedness: fraction of specific factual claims in ANSWER supported by CONTEXT (1=fully grounded)
hallucination: fraction of claims that contradict or invent facts not in CONTEXT (0=no hallucination)
retrieval_precision: how precisely the context addresses the question (1=perfectly relevant)

QUESTION: {question}

ANSWER: {answer}

CONTEXT (what was retrieved and sent to the LLM):
{context}

Reply ONLY with the JSON object.
""".strip()

_MANUAL_JUDGE_PROMPT = """
Evaluate this answer and return JSON with exactly these keys:

{{
  "accuracy": <0.0|0.5|1.0>,
  "accuracy_reason": "<one sentence>",
  "groundedness": <0.0-1.0>,
  "groundedness_reason": "<one sentence>",
  "hallucination": <0.0-1.0>,
  "hallucination_reason": "<one sentence>",
  "retrieval_precision": <0.0-1.0>,
  "retrieval_precision_reason": "<one sentence>",
  "recall": <0.0-1.0>,
  "recall_reason": "<one sentence>"
}}

accuracy: 1.0=correctly addresses expected; 0.5=partially; 0.0=wrong/missing
groundedness: fraction of claims supported by CONTEXT
hallucination: fraction of claims that contradict CONTEXT (0=none)
retrieval_precision: how precisely the context targets the question
recall: fraction of key facts from EXPECTED ANSWER that appear in ACTUAL ANSWER

QUESTION: {question}
EXPECTED ANSWER: {expected}
ACTUAL ANSWER: {answer}
CONTEXT: {context}

Reply ONLY with the JSON object.
""".strip()


def _call_judge(settings, prompt: str) -> dict:
    try:
        resp = requests.post(
            f"{settings.ollama_host}/api/generate",
            json={
                "model": settings.text_model,
                "system": _JUDGE_SYSTEM,
                "prompt": prompt,
                "stream": False,
                "options": {"num_ctx": 2048, "num_predict": 400, "temperature": 0.0},
            },
            timeout=(10, 90),
        )
        resp.raise_for_status()
        raw = resp.json().get("response", "").strip()
        raw = re.sub(r"```(?:json)?", "", raw).strip().rstrip("`").strip()
        return json.loads(raw)
    except Exception as e:
        return {"_error": str(e)}


# ── Auto-eval from logs ───────────────────────────────────────────────────────

def load_ask_logs(n: int = 200) -> list[dict]:
    """Read last N ask/ask_captions entries from ollama_calls.jsonl."""
    try:
        if not _LOG_FILE.exists():
            return []
        lines = _LOG_FILE.read_text(encoding="utf-8").splitlines()
        entries = []
        for line in reversed(lines):
            if not line.strip():
                continue
            try:
                e = json.loads(line)
                if e.get("call_type") in ("ask", "ask_captions") and e.get("status") == "ok":
                    entries.append(e)
                    if len(entries) >= n:
                        break
            except Exception:
                continue
        return entries
    except Exception:
        return []


def score_log_entry(entry: dict, skip_judge: bool = False) -> dict:
    """Score a single log entry with auto metrics (no ground truth)."""
    settings = get_settings()
    prompt   = entry.get("prompt", "")
    response = entry.get("response", "")
    elapsed  = entry.get("elapsed_ms", 0)
    ts       = entry.get("ts", "")
    model    = entry.get("model", "")

    question = _extract_question(prompt)
    context  = _extract_context(prompt)

    score_lat  = _score_latency(elapsed)
    score_grnd = -1.0
    score_hall = -1.0
    score_prec = -1.0
    judge_notes = {}

    if not skip_judge and response and question:
        p = _AUTO_JUDGE_PROMPT.format(
            question=question,
            answer=response[:1000],
            context=context[:2000],
        )
        j = _call_judge(settings, p)
        if "_error" not in j:
            score_grnd = float(j.get("groundedness", -1))
            raw_hall   = float(j.get("hallucination", -1))
            score_hall = (1.0 - raw_hall) if raw_hall >= 0 else -1.0
            score_prec = float(j.get("retrieval_precision", -1))
            judge_notes = {
                "groundedness":        j.get("groundedness_reason", ""),
                "hallucination":       j.get("hallucination_reason", ""),
                "retrieval_precision": j.get("retrieval_precision_reason", ""),
            }

    return {
        "ts":              ts,
        "model":           model,
        "question":        question,
        "answer":          response,
        "context":         context[:3000],
        "elapsed_ms":      elapsed,
        "score_latency":   round(score_lat,  3),
        "score_groundedness":  round(score_grnd, 3) if score_grnd >= 0 else -1,
        "score_hallucination": round(score_hall, 3) if score_hall >= 0 else -1,
        "score_precision": round(score_prec, 3) if score_prec >= 0 else -1,
        "score_accuracy":  -1,
        "score_recall":    -1,
        "judge_notes":     judge_notes,
        "mode":            "auto",
    }


# ── Manual eval (EvalCase) ────────────────────────────────────────────────────

def run_case(db, case_id, skip_judge: bool = False) -> dict:
    """Run a manual EvalCase through QAEngine and score all metrics."""
    from app.storage.eval_models import EvalCase
    from app.rag.qa_engine import QAEngine

    settings = get_settings()
    logger   = get_logger()

    case = db.query(EvalCase).filter(EvalCase.id == case_id).first()
    if not case:
        return {"error": "case_not_found"}

    engine = QAEngine(db)
    t0 = time.monotonic()
    try:
        result = engine.ask(
            question=case.question,
            video_filename=case.video_filename or None,
        )
        elapsed_ms = (time.monotonic() - t0) * 1000
    except Exception as e:
        elapsed_ms = (time.monotonic() - t0) * 1000
        return {
            "case_id": str(case_id), "actual_answer": None,
            "context_sent": "", "sources": [], "fast_path": False,
            "latency_ms": elapsed_ms, "error": str(e)[:200],
            "score_accuracy": -1, "score_groundedness": -1,
            "score_hallucination": -1, "score_latency": _score_latency(elapsed_ms),
            "score_precision": -1, "score_recall": -1,
            "judge_notes": {}, "model_name": settings.text_model,
        }

    actual  = result.get("answer", "")
    sources = result.get("sources", [])
    is_fast = result.get("fast_path", False)
    context = result.get("context", "")

    scores = {
        "accuracy": -1, "groundedness": -1,
        "hallucination_raw": -1, "precision": -1, "recall": -1,
    }
    judge_notes = {}

    if not skip_judge and actual:
        p = _MANUAL_JUDGE_PROMPT.format(
            question=case.question,
            expected=case.expected,
            answer=actual[:1500],
            context=context[:2000],
        )
        j = _call_judge(settings, p)
        if "_error" not in j:
            scores["accuracy"]        = float(j.get("accuracy", -1))
            scores["groundedness"]    = float(j.get("groundedness", -1))
            scores["hallucination_raw"] = float(j.get("hallucination", -1))
            scores["precision"]       = float(j.get("retrieval_precision", -1))
            scores["recall"]          = float(j.get("recall", -1))
            judge_notes = {
                "accuracy":            j.get("accuracy_reason", ""),
                "groundedness":        j.get("groundedness_reason", ""),
                "hallucination":       j.get("hallucination_reason", ""),
                "retrieval_precision": j.get("retrieval_precision_reason", ""),
                "recall":              j.get("recall_reason", ""),
            }

    hall_raw = scores["hallucination_raw"]
    score_hall = (1.0 - hall_raw) if hall_raw >= 0 else -1.0

    return {
        "case_id":      str(case_id),
        "actual_answer": actual,
        "context_sent": context[:4000],
        "sources":      [{k: v for k, v in s.items() if k != "best_crop_path"} for s in sources],
        "fast_path":    bool(is_fast),
        "latency_ms":   round(elapsed_ms, 1),
        "error":        None,
        "score_accuracy":     scores["accuracy"],
        "score_groundedness": scores["groundedness"],
        "score_hallucination": score_hall,
        "score_latency":      _score_latency(elapsed_ms),
        "score_precision":    scores["precision"],
        "score_recall":       scores["recall"],
        "judge_notes":        judge_notes,
        "model_name":         settings.text_model,
    }


# ── Aggregate stats ───────────────────────────────────────────────────────────

def aggregate(scored: list[dict]) -> dict:
    """Compute aggregate statistics over a list of scored entries."""
    def _mean(key):
        vals = [s[key] for s in scored if s.get(key, -1) >= 0]
        return round(statistics.mean(vals), 3) if vals else None

    def _pct(vals_ms, p):
        v = sorted(x for x in vals_ms if x)
        if not v:
            return None
        return round(v[int(len(v) * p / 100)], 1)

    lats = [s["elapsed_ms"] for s in scored if s.get("elapsed_ms")]
    return {
        "n":                 len(scored),
        "accuracy":          _mean("score_accuracy"),
        "groundedness":      _mean("score_groundedness"),
        "hallucination":     _mean("score_hallucination"),
        "latency_score":     _mean("score_latency"),
        "precision":         _mean("score_precision"),
        "recall":            _mean("score_recall"),
        "latency_p50_ms":    _pct(lats, 50),
        "latency_p95_ms":    _pct(lats, 95),
        "latency_mean_ms":   round(sum(lats)/len(lats), 1) if lats else None,
    }