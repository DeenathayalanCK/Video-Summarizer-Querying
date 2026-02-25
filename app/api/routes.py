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
from app.core.logging import get_logger

router = APIRouter()
logger = get_logger()


# ── Dependency ────────────────────────────────────────────────────────────────

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# ── Request / Response schemas ────────────────────────────────────────────────

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


# ── Routes ────────────────────────────────────────────────────────────────────

@router.get("/health")
def health():
    return {"status": "ok"}


@router.post("/ask", response_model=AskResponse)
def ask(body: AskRequest, db: Session = Depends(get_db)):
    """
    Ask a natural language question over all ingested video captions.
    Optionally scope to a specific video, camera, or time range.
    """
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
    """
    Semantic search over captions. Returns matching moments with similarity scores.
    No LLM call — fast retrieval only.
    """
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
    """List all ingested videos with caption counts."""
    from sqlalchemy import func
    rows = (
        db.query(Caption.video_filename, func.count(Caption.id).label("caption_count"))
        .group_by(Caption.video_filename)
        .order_by(Caption.video_filename)
        .all()
    )
    return {
        "videos": [
            {"video_filename": r.video_filename, "caption_count": r.caption_count}
            for r in rows
        ]
    }


@router.get("/timeline/{video_filename}")
def get_timeline(video_filename: str, db: Session = Depends(get_db)):
    """Return all captions for a video in chronological order."""
    retriever = CaptionRetriever(db)
    timeline = retriever.get_timeline(video_filename)
    if not timeline:
        raise HTTPException(status_code=404, detail="Video not found or no captions yet")
    return {"video_filename": video_filename, "count": len(timeline), "timeline": timeline}


@router.get("/keyframe")
def get_keyframe(path: str = Query(..., description="keyframe_path from caption")):
    """Serve a keyframe image by its stored path."""
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Keyframe not found")
    return FileResponse(path, media_type="image/jpeg")


@router.post("/index")
def trigger_reindex(db: Session = Depends(get_db)):
    """
    Manually trigger embedding of any captions that haven't been indexed yet.
    Useful after adding new videos without restarting the pipeline.
    """
    indexer = CaptionIndexer(db)
    count = indexer.index_all_unindexed()
    return {"indexed": count}