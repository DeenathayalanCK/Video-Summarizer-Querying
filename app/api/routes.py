from fastapi import APIRouter, Depends, Query, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
from sqlalchemy.orm import Session
from typing import Optional
import os

from app.storage.database import SessionLocal
from app.storage.models import Caption
from app.rag.qa_engine import QAEngine
from app.rag.retriever import CaptionRetriever
from app.rag.indexer import CaptionIndexer
from app.rag.summarizer import VideoSummarizer
from app.storage.repository import EventRepository
from app.core.logging import get_logger

router = APIRouter()
logger = get_logger()


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# ── Schemas ────────────────────────────────────────────────────────────────────

class AskRequest(BaseModel):
    question: str
    video_filename: Optional[str] = None
    camera_id: Optional[str] = None
    min_second: Optional[float] = None
    max_second: Optional[float] = None


class AskResponse(BaseModel):
    question: str
    answer: str
    sources: list[dict]


class SummaryResponse(BaseModel):
    video_filename: str
    camera_id: str
    summary: str
    caption_count: int
    duration_seconds: float
    model_name: str
    created_at: str
    updated_at: str


class ProcessingStatusResponse(BaseModel):
    video_filename: str
    status: str
    scenes_detected: int
    scenes_captioned: int
    total_frames_sampled: Optional[int]
    current_second: Optional[float]
    progress_pct: Optional[float]
    error_count: int
    last_error: Optional[str]
    started_at: Optional[str]
    completed_at: Optional[str]
    updated_at: Optional[str]


# ── Health ─────────────────────────────────────────────────────────────────────

@router.get("/health")
def health():
    return {"status": "ok"}


# ── Q&A ────────────────────────────────────────────────────────────────────────

@router.post("/ask", response_model=AskResponse)
def ask(body: AskRequest, db: Session = Depends(get_db)):
    """Ask a natural language question grounded in video captions."""
    engine = QAEngine(db)
    result = engine.ask(
        question=body.question,
        video_filename=body.video_filename,
        camera_id=body.camera_id,
        min_second=body.min_second,
        max_second=body.max_second,
    )
    return AskResponse(
        question=body.question,
        answer=result["answer"],
        sources=result["sources"],
    )


# ── Search ─────────────────────────────────────────────────────────────────────

@router.get("/search")
def search(
    q: str = Query(...),
    video_filename: Optional[str] = Query(None),
    camera_id: Optional[str] = Query(None),
    min_second: Optional[float] = Query(None),
    max_second: Optional[float] = Query(None),
    top_k: int = Query(8, ge=1, le=50),
    db: Session = Depends(get_db),
):
    retriever = CaptionRetriever(db, top_k=top_k)
    results = retriever.search(
        query=q, video_filename=video_filename,
        camera_id=camera_id, min_second=min_second, max_second=max_second,
    )
    return {"query": q, "count": len(results), "results": results}


# ── Videos ────────────────────────────────────────────────────────────────────

@router.get("/videos")
def list_videos(db: Session = Depends(get_db)):
    """List all ingested videos with caption counts, summary status, and processing status."""
    from sqlalchemy import func
    from app.storage.models import VideoSummary, ProcessingStatus

    rows = (
        db.query(Caption.video_filename, func.count(Caption.id).label("caption_count"))
        .group_by(Caption.video_filename)
        .order_by(Caption.video_filename)
        .all()
    )
    summaries = {s.video_filename for s in db.query(VideoSummary.video_filename).all()}
    statuses = {
        s.video_filename: s.status
        for s in db.query(ProcessingStatus).all()
    }

    return {
        "videos": [
            {
                "video_filename": r.video_filename,
                "caption_count": r.caption_count,
                "has_summary": r.video_filename in summaries,
                "processing_status": statuses.get(r.video_filename, "unknown"),
            }
            for r in rows
        ]
    }


# ── Timeline ──────────────────────────────────────────────────────────────────

@router.get("/timeline/{video_filename}")
def get_timeline(video_filename: str, db: Session = Depends(get_db)):
    retriever = CaptionRetriever(db)
    timeline = retriever.get_timeline(video_filename)
    if not timeline:
        raise HTTPException(status_code=404, detail="Video not found or no captions yet")
    return {"video_filename": video_filename, "count": len(timeline), "timeline": timeline}


# ── Summary ───────────────────────────────────────────────────────────────────

@router.get("/summary/{video_filename}", response_model=SummaryResponse)
def get_summary(video_filename: str, db: Session = Depends(get_db)):
    from app.core.config import get_settings
    repo = EventRepository(db)
    summary = repo.get_summary(video_filename, get_settings().camera_id)
    if not summary:
        raise HTTPException(
            status_code=404,
            detail="No summary found. POST /api/v1/summarize/{video_filename} to generate."
        )
    return SummaryResponse(
        video_filename=summary.video_filename,
        camera_id=summary.camera_id,
        summary=summary.summary_text,
        caption_count=summary.caption_count,
        duration_seconds=summary.duration_seconds,
        model_name=summary.model_name,
        created_at=summary.created_at.isoformat(),
        updated_at=summary.updated_at.isoformat() if summary.updated_at else summary.created_at.isoformat(),
    )


@router.post("/summarize/{video_filename}")
def generate_summary(
    video_filename: str,
    force: bool = Query(False),
    db: Session = Depends(get_db),
):
    summarizer = VideoSummarizer(db)
    try:
        summary = summarizer.summarize(video_filename, force=force)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    return {
        "video_filename": summary.video_filename,
        "caption_count": summary.caption_count,
        "duration_seconds": summary.duration_seconds,
        "model_name": summary.model_name,
        "summary": summary.summary_text,
    }


@router.post("/summarize-all")
def summarize_all(force: bool = Query(False), db: Session = Depends(get_db)):
    results = VideoSummarizer(db).summarize_all(force=force)
    return {"summarized": len(results), "videos": [r.video_filename for r in results]}


@router.get("/summaries")
def list_summaries(db: Session = Depends(get_db)):
    repo = EventRepository(db)
    summaries = repo.list_summaries()
    return {
        "count": len(summaries),
        "summaries": [
            {
                "video_filename": s.video_filename,
                "camera_id": s.camera_id,
                "caption_count": s.caption_count,
                "duration_seconds": s.duration_seconds,
                "model_name": s.model_name,
                "created_at": s.created_at.isoformat(),
            }
            for s in summaries
        ],
    }


# ── Processing Status API ─────────────────────────────────────────────────────

def _status_to_response(s) -> dict:
    """Convert a ProcessingStatus row to API response dict with progress %."""
    pct = None
    if s.total_frames_sampled and s.total_frames_sampled > 0 and s.current_second is not None:
        # Estimate: scenes_captioned / scenes_detected gives caption progress
        # Use current_second / estimated_duration for frame-level progress
        if s.scenes_detected and s.scenes_detected > 0:
            pct = round((s.scenes_captioned / s.scenes_detected) * 100, 1)

    return {
        "video_filename": s.video_filename,
        "status": s.status,
        "scenes_detected": s.scenes_detected,
        "scenes_captioned": s.scenes_captioned,
        "total_frames_sampled": s.total_frames_sampled,
        "current_second": s.current_second,
        "progress_pct": pct,
        "error_count": s.error_count,
        "last_error": s.last_error,
        "started_at": s.started_at.isoformat() if s.started_at else None,
        "completed_at": s.completed_at.isoformat() if s.completed_at else None,
        "updated_at": s.updated_at.isoformat() if s.updated_at else None,
    }


@router.get("/status")
def list_processing_status(db: Session = Depends(get_db)):
    """
    Returns processing status for all known videos.
    Poll this to track pipeline progress in real time.
    """
    repo = EventRepository(db)
    statuses = repo.list_statuses()
    return {
        "count": len(statuses),
        "statuses": [_status_to_response(s) for s in statuses],
    }


@router.get("/status/{video_filename}")
def get_processing_status(video_filename: str, db: Session = Depends(get_db)):
    """
    Returns detailed processing status for a single video.
    Includes progress %, current second, error count, timing.
    """
    repo = EventRepository(db)
    status = repo.get_status(video_filename)
    if not status:
        raise HTTPException(
            status_code=404,
            detail="No processing record found. Video may not have been queued yet."
        )
    return _status_to_response(status)


@router.delete("/status/{video_filename}")
def reset_processing_status(video_filename: str, db: Session = Depends(get_db)):
    """
    Delete the processing status for a video, allowing it to be reprocessed.
    Use when a video is stuck in 'failed' or 'running' state.
    """
    repo = EventRepository(db)
    status = repo.get_status(video_filename)
    if not status:
        raise HTTPException(status_code=404, detail="No processing record found.")
    db.delete(status)
    db.commit()
    return {"deleted": video_filename, "message": "Video will be reprocessed on next pipeline run."}


# ── Keyframe + Reindex ────────────────────────────────────────────────────────

@router.get("/keyframe")
def get_keyframe(path: str = Query(...)):
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Keyframe not found")
    return FileResponse(path, media_type="image/jpeg")


@router.post("/index")
def trigger_reindex(db: Session = Depends(get_db)):
    count = CaptionIndexer(db).index_all_unindexed()
    return {"indexed": count}