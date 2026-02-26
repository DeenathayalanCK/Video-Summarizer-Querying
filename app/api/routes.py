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


# ── Dependency ─────────────────────────────────────────────────────────────────

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


# ── Routes ─────────────────────────────────────────────────────────────────────

@router.get("/health")
def health():
    return {"status": "ok"}


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


@router.get("/search")
def search(
    q: str = Query(..., description="Natural language search query"),
    video_filename: Optional[str] = Query(None),
    camera_id: Optional[str] = Query(None),
    min_second: Optional[float] = Query(None),
    max_second: Optional[float] = Query(None),
    top_k: int = Query(8, ge=1, le=50),
    db: Session = Depends(get_db),
):
    """Semantic search — fast retrieval, no LLM call."""
    retriever = CaptionRetriever(db, top_k=top_k)
    results = retriever.search(
        query=q,
        video_filename=video_filename,
        camera_id=camera_id,
        min_second=min_second,
        max_second=max_second,
    )
    return {"query": q, "count": len(results), "results": results}


@router.get("/videos")
def list_videos(db: Session = Depends(get_db)):
    """List all ingested videos with caption counts and summary availability."""
    from sqlalchemy import func
    from app.storage.models import VideoSummary

    rows = (
        db.query(Caption.video_filename, func.count(Caption.id).label("caption_count"))
        .group_by(Caption.video_filename)
        .order_by(Caption.video_filename)
        .all()
    )

    # Check which have summaries
    summaries = {
        s.video_filename: True
        for s in db.query(VideoSummary.video_filename).all()
    }

    return {
        "videos": [
            {
                "video_filename": r.video_filename,
                "caption_count": r.caption_count,
                "has_summary": summaries.get(r.video_filename, False),
            }
            for r in rows
        ]
    }


@router.get("/timeline/{video_filename}")
def get_timeline(video_filename: str, db: Session = Depends(get_db)):
    """All captions for a video in chronological order."""
    retriever = CaptionRetriever(db)
    timeline = retriever.get_timeline(video_filename)
    if not timeline:
        raise HTTPException(status_code=404, detail="Video not found or no captions yet")
    return {"video_filename": video_filename, "count": len(timeline), "timeline": timeline}


@router.get("/summary/{video_filename}", response_model=SummaryResponse)
def get_summary(video_filename: str, db: Session = Depends(get_db)):
    """
    Get the stored summary for a video.
    Returns 404 if not yet summarized — call POST /summarize/{video_filename} first.
    """
    repo = EventRepository(db)
    from app.core.config import get_settings
    settings = get_settings()

    summary = repo.get_summary(video_filename, settings.camera_id)
    if not summary:
        raise HTTPException(
            status_code=404,
            detail="No summary found. Run POST /api/v1/summarize/{video_filename} to generate one."
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
    force: bool = Query(False, description="Force regenerate even if summary exists"),
    db: Session = Depends(get_db),
):
    """
    Trigger summary generation for a specific video.
    Uses cached version unless force=true or new captions were added.
    """
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
def summarize_all(
    force: bool = Query(False, description="Force regenerate all summaries"),
    db: Session = Depends(get_db),
):
    """Generate summaries for all videos that have captions."""
    summarizer = VideoSummarizer(db)
    results = summarizer.summarize_all(force=force)
    return {
        "summarized": len(results),
        "videos": [r.video_filename for r in results],
    }


@router.get("/summaries")
def list_summaries(db: Session = Depends(get_db)):
    """List all stored summaries with metadata."""
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


@router.get("/keyframe")
def get_keyframe(path: str = Query(..., description="keyframe_path from caption")):
    """Serve a keyframe image by its stored path."""
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Keyframe not found")
    return FileResponse(path, media_type="image/jpeg")


@router.post("/index")
def trigger_reindex(db: Session = Depends(get_db)):
    """Manually embed any captions that haven't been indexed yet."""
    indexer = CaptionIndexer(db)
    count = indexer.index_all_unindexed()
    return {"indexed": count}