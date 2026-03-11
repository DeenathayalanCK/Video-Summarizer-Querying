from fastapi import APIRouter, Depends, Query, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, ConfigDict
from sqlalchemy.orm import Session
from typing import Optional
import os

from app.storage.database import SessionLocal
from app.storage.models import Caption, DetectedObject, TrackEvent
from app.rag.qa_engine import QAEngine
from app.rag.retriever import CaptionRetriever
from app.rag.object_retriever import ObjectRetriever
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
    model_config = ConfigDict(protected_namespaces=())
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

@router.get("/mode")
def get_mode():
    """
    Returns whether the system is in live (RTSP) or batch (file) mode.
    Used by the UI to adapt the sidebar and show/hide tabs.
    """
    from app.core.config import get_settings
    s = get_settings()
    return {
        "mode":   "live" if s.is_rtsp else "batch",
        "source": s.video_input_path,
        "camera_id": s.camera_id,
        "live_window_minutes": s.live_window_minutes if s.is_rtsp else None,
    }


@router.post("/ask", response_model=AskResponse)
def ask(body: AskRequest, db: Session = Depends(get_db)):
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


@router.post("/ask-stream")
def ask_stream(body: AskRequest, db: Session = Depends(get_db)):
    """
    Streaming version of /ask. Returns a Server-Sent Events stream.
    Each event is: data: {"token": "..."} or data: {"done": true, "sources": [...]}
    Fast-path queries (plate, count, presence, behaviour) return immediately
    without calling the LLM at all.

    The Ollama semaphore from WindowManager is acquired for the LLM portion only.
    Fast-path answers bypass it entirely (no Ollama call).
    This prevents QA (llama3.2 @ 4096 KV) overlapping with postprocessing
    (summary/attributes @ 2048 KV), which would saturate CPU RAM and cause timeouts.
    """
    engine = QAEngine(db)

    def generate():
        # Fast-path: no Ollama call needed — stream immediately without sem
        from app.rag.fast_path import try_fast_path
        import json as _json
        fast = try_fast_path(db, body.question, body.video_filename)
        if fast.get("answered"):
            yield f"data: {_json.dumps({'token': fast['answer']})}\n\n"
            yield f"data: {_json.dumps({'done': True, 'sources': fast.get('sources', []), 'fast_path': True})}\n\n"
            return

        # LLM path: signal priority then acquire semaphore.
        # Setting _ask_pending causes the pipeline to skip its NEXT Ollama step
        # (attrs/activity/summary) so the user ask gets Ollama access quickly.
        from app.vision.window_manager import WindowManager
        wm = WindowManager.get_instance()
        _has_priority = hasattr(wm, '_ask_pending')
        if _has_priority:
            wm._ask_pending.set()
        try:
            wm._ollama_sem.acquire(blocking=True)
            if _has_priority:
                wm._ask_pending.clear()
            try:
                    yield from engine.stream_ask(
                question=body.question,
                video_filename=body.video_filename,
                camera_id=body.camera_id,
                min_second=body.min_second,
                max_second=body.max_second,
            )
            finally:
                wm._ollama_sem.release()
        finally:
            if _has_priority:
                wm._ask_pending.clear()

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",   # disable nginx buffering
        },
    )


# ── Legacy caption search ──────────────────────────────────────────────────────

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


# ── Phase 6A: Object detection search ─────────────────────────────────────────

@router.get("/search/objects")
def search_objects(
    q: str = Query(..., description="Natural language query, e.g. 'person detected'"),
    video_filename: Optional[str] = Query(None),
    camera_id: Optional[str] = Query(None),
    object_class: Optional[str] = Query(None, description="Filter: person, car, truck, bus, motorcycle, bicycle"),
    min_second: Optional[float] = Query(None),
    max_second: Optional[float] = Query(None),
    top_k: int = Query(8, ge=1, le=50),
    db: Session = Depends(get_db),
):
    """Search individual frame-level detections semantically."""
    retriever = ObjectRetriever(db, top_k=top_k)
    results = retriever.search_detections(
        query=q,
        video_filename=video_filename,
        camera_id=camera_id,
        object_class=object_class,
        min_second=min_second,
        max_second=max_second,
    )
    return {"query": q, "count": len(results), "results": results}


@router.get("/search/events")
def search_track_events(
    q: str = Query(..., description="Natural language query, e.g. 'vehicle entered'"),
    video_filename: Optional[str] = Query(None),
    camera_id: Optional[str] = Query(None),
    event_type: Optional[str] = Query(None, description="Filter: entry, exit, dwell"),
    object_class: Optional[str] = Query(None),
    top_k: int = Query(8, ge=1, le=50),
    db: Session = Depends(get_db),
):
    """Search track lifecycle events (entry/exit/dwell) semantically."""
    retriever = ObjectRetriever(db, top_k=top_k)

    # Semantic search
    semantic = retriever.search_track_events(
        query=q, video_filename=video_filename,
        camera_id=camera_id, event_type=event_type, object_class=object_class,
    )

    # Attribute keyword search — catches color/type/clothing queries semantic misses
    keyword = retriever.attribute_keyword_search(
        query=q, video_filename=video_filename,
        event_type=event_type, object_class=object_class, top_k=top_k,
    )

    # Merge: keyword hits first, then semantic hits not already present
    seen = {r["event_id"] for r in keyword}
    merged = keyword[:]
    for r in semantic:
        if r["event_id"] not in seen:
            merged.append(r)
            seen.add(r["event_id"])
    for r in merged:
        if "match_type" not in r:
            r["match_type"] = "semantic"

    # Deduplicate by track_id: one physical object = one search card.
    # A car generates entry+exit+dwell rows (different event_ids but same track).
    # Keep the highest-scoring event; attach others as also_events for UI context.
    track_best: dict[str, dict] = {}
    for r in merged:
        k = f"{r['video_filename']}:{r['track_id']}"
        if k not in track_best:
            track_best[k] = {**r, "also_events": []}
        elif r["score"] > track_best[k]["score"]:
            prev = {kk: vv for kk, vv in track_best[k].items() if kk != "also_events"}
            also = track_best[k]["also_events"]
            track_best[k] = {**r, "also_events": also + [prev]}
        else:
            track_best[k]["also_events"].append(r)

    results = list(track_best.values())[:top_k]
    return {"query": q, "count": len(results), "results": results}


# ── Phase 6A: Videos list ─────────────────────────────────────────────────────

@router.get("/videos")
def list_videos(db: Session = Depends(get_db)):
    """List all videos with detection stats and processing status."""
    from sqlalchemy import func
    from app.storage.models import VideoSummary, ProcessingStatus

    # Detection counts (Phase 6A data)
    det_counts = dict(
        db.query(DetectedObject.video_filename, func.count(DetectedObject.id))
        .group_by(DetectedObject.video_filename)
        .all()
    )

    # Unique track counts
    track_counts = dict(
        db.query(TrackEvent.video_filename, func.count(TrackEvent.id))
        .filter(TrackEvent.event_type == "entry")
        .group_by(TrackEvent.video_filename)
        .all()
    )

    # Legacy caption counts
    caption_counts = dict(
        db.query(Caption.video_filename, func.count(Caption.id))
        .group_by(Caption.video_filename)
        .all()
    )

    summaries = {s.video_filename for s in db.query(VideoSummary.video_filename).all()}
    statuses = {
        s.video_filename: s.status
        for s in db.query(ProcessingStatus).all()
    }
    phase_6b = {
        s.video_filename: bool(s.phase_6b_completed)
        for s in db.query(ProcessingStatus).all()
    }

    # Union of all known video filenames
    all_videos = set(det_counts) | set(caption_counts) | set(statuses)

    from app.core.config import get_settings
    is_live = get_settings().is_rtsp

    def _parse_window_label(vf: str) -> dict:
        """For live window keys, extract human-readable display label."""
        if not is_live:
            return {"display_label": vf, "is_window": False}
        try:
            parts = vf.rsplit("_", 2)
            if len(parts) == 3:
                from datetime import datetime
                dt = datetime.strptime(parts[1] + parts[2], "%Y%m%d%H%M")
                from app.core.config import get_settings as _gs
                mins = _gs().live_window_minutes
                end_dt = dt.replace(minute=dt.minute + mins) if dt.minute + mins < 60 else dt
                label = f"{dt.strftime('%b %d  %H:%M')} – {(dt.minute+mins)%60:02d}"
                return {"display_label": label, "is_window": True,
                        "window_start": dt.isoformat()}
        except Exception:
            pass
        return {"display_label": vf, "is_window": False}

    videos_list = []
    for vf in sorted(all_videos, reverse=is_live):  # newest first in live mode
        info = _parse_window_label(vf)
        videos_list.append({
            "video_filename":    vf,
            "display_label":     info["display_label"],
            "is_window":         info.get("is_window", False),
            "window_start":      info.get("window_start"),
            "detection_count":   det_counts.get(vf, 0),
            "unique_tracks":     track_counts.get(vf, 0),
            "caption_count":     caption_counts.get(vf, 0),
            "has_summary":       vf in summaries,
            "processing_status": statuses.get(vf, "unknown"),
            "pipeline":          "detection" if (vf in det_counts or vf in statuses)
                                 else "caption" if vf in caption_counts else "unknown",
            "phase_6b_completed": phase_6b.get(vf, False),
        })
    return {"videos": videos_list, "mode": "live" if is_live else "batch"}


# ── Phase 6A: Track timeline ───────────────────────────────────────────────────

@router.get("/timeline/{video_filename}")
def get_timeline(video_filename: str, db: Session = Depends(get_db)):
    """
    Returns track events for a video in chronological order.
    Falls back to caption timeline if no detection data exists.
    """
    repo = EventRepository(db)

    if repo.has_detection_data(video_filename):
        retriever = ObjectRetriever(db)
        timeline = retriever.get_track_timeline(video_filename)
        if timeline:
            return {
                "video_filename": video_filename,
                "pipeline": "detection",
                "count": len(timeline),
                "timeline": timeline,
            }

    # Fallback to legacy caption timeline
    retriever = CaptionRetriever(db)
    timeline = retriever.get_timeline(video_filename)
    if not timeline:
        raise HTTPException(status_code=404, detail="Video not found or no data yet")

    return {
        "video_filename": video_filename,
        "pipeline": "caption",
        "count": len(timeline),
        "timeline": timeline,
    }


# ── Phase 6A: Detections for a video ──────────────────────────────────────────

@router.get("/detections/{video_filename}")
def get_detections(
    video_filename: str,
    object_class: Optional[str] = Query(None),
    track_id: Optional[int] = Query(None),
    min_second: Optional[float] = Query(None),
    max_second: Optional[float] = Query(None),
    db: Session = Depends(get_db),
):
    """
    Return all detected objects for a video with optional filters.
    Enriches each detection with its TrackEvent attributes (gender, clothing,
    activity_hint, objects_nearby, face data) by joining on track_id.
    """
    from app.storage.models import TrackEvent
    repo = EventRepository(db)
    objects = repo.get_detected_objects(
        video_filename,
        object_class=object_class,
        track_id=track_id,
        min_second=min_second,
        max_second=max_second,
    )

    # Build track_id → entry TrackEvent attributes lookup (one DB query)
    track_ids = list({o.track_id for o in objects if o.track_id is not None})
    track_attrs: dict = {}
    track_crops: dict = {}
    if track_ids:
        entry_events = (
            db.query(TrackEvent)
            .filter(
                TrackEvent.video_filename == video_filename,
                TrackEvent.event_type == "entry",
                TrackEvent.track_id.in_(track_ids),
            )
            .all()
        )
        for ev in entry_events:
            if ev.attributes:
                attrs = dict(ev.attributes)
                attrs.setdefault("object_class", ev.object_class)
                track_attrs[ev.track_id] = attrs
            track_crops[ev.track_id] = ev.best_crop_path

    return {
        "video_filename": video_filename,
        "count": len(objects),
        "detections": [
            {
                "id": str(o.id),
                "second": o.frame_second_offset,
                "object_class": o.object_class,
                "confidence": round(o.confidence, 3),
                "track_id": o.track_id,
                "quadrant": o.frame_quadrant,
                "crop_path": o.crop_path,
                # Best crop from TrackEvent (higher quality than individual detection)
                "best_crop_path": track_crops.get(o.track_id),
                "bbox": {
                    "x1": o.bbox_x1, "y1": o.bbox_y1,
                    "x2": o.bbox_x2, "y2": o.bbox_y2,
                },
                # Attributes from TrackEvent (null until attribute extraction runs)
                "attributes": track_attrs.get(o.track_id),
            }
            for o in objects
        ],
    }


@router.get("/tracks/{video_filename}")
def get_tracks(
    video_filename: str,
    event_type: Optional[str] = Query(None),
    object_class: Optional[str] = Query(None),
    db: Session = Depends(get_db),
):
    """Return all track events for a video."""
    repo = EventRepository(db)
    events = repo.get_track_events(
        video_filename, event_type=event_type, object_class=object_class,
    )
    return {
        "video_filename": video_filename,
        "count": len(events),
        "track_events": [
            {
                "id": str(e.id),
                "track_id": e.track_id,
                "object_class": e.object_class,
                "event_type": e.event_type,
                "first_seen": e.first_seen_second,
                "last_seen": e.last_seen_second,
                "duration": e.duration_seconds,
                "best_frame_second": e.best_frame_second,
                "best_crop_path": e.best_crop_path,
                "best_confidence": round(e.best_confidence or 0, 3),
                "attributes": e.attributes,
                # Convenience top-level fields surfaced from attributes
                # (avoids UI having to dig into the attributes blob)
                "face_detected":   (e.attributes or {}).get("face_detected", False),
                "face_crop_path":  (e.attributes or {}).get("face_crop_path"),
                "face_confidence": (e.attributes or {}).get("face_confidence", 0.0),
                "person_label":    (e.attributes or {}).get("person_label"),
                "entry_wall_time": (e.attributes or {}).get("entry_wall_time"),
                "exit_wall_time":  (e.attributes or {}).get("exit_wall_time"),
                "activity_hint":   (e.attributes or {}).get("activity_hint"),
                "objects_nearby":  (e.attributes or {}).get("objects_nearby", []),
            }
            for e in events
        ],
    }


# ── Summary ────────────────────────────────────────────────────────────────────

@router.get("/summary/{video_filename}", response_model=SummaryResponse)
def get_summary(video_filename: str, db: Session = Depends(get_db)):
    from app.core.config import get_settings
    repo = EventRepository(db)
    summary = repo.get_summary(video_filename, get_settings().camera_id)
    if not summary:
        raise HTTPException(
            status_code=404,
            detail="No summary found. Process the video first.",
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
    repo = EventRepository(db)
    try:
        # Use track-based summarizer if detection data exists
        if repo.has_detection_data(video_filename):
            summary = summarizer.summarize_from_tracks(video_filename, force=force)
        else:
            summary = summarizer.summarize(video_filename, force=force)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        # Catches requests.exceptions.Timeout, ConnectionError, etc.
        import requests as _req
        if isinstance(e, (_req.exceptions.Timeout, _req.exceptions.ConnectionError)):
            raise HTTPException(status_code=503,
                                detail=f"Ollama unavailable or timed out: {e}")
        raise HTTPException(status_code=500, detail=f"Summary generation failed: {e}")
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


# ── Processing Status ──────────────────────────────────────────────────────────

def _status_to_response(s) -> dict:
    pct = None
    if s.total_frames_sampled and s.total_frames_sampled > 0 and s.current_second is not None:
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
        "phase_6b_completed": bool(s.phase_6b_completed),
        "phase_6b_tracks_attributed": s.phase_6b_tracks_attributed,
        "current_step": getattr(s, "current_step", None),
    }


@router.get("/status")
def list_processing_status(db: Session = Depends(get_db)):
    repo = EventRepository(db)
    statuses = repo.list_statuses()
    return {"count": len(statuses), "statuses": [_status_to_response(s) for s in statuses]}


@router.get("/status/{video_filename}")
def get_processing_status(video_filename: str, db: Session = Depends(get_db)):
    repo = EventRepository(db)
    status = repo.get_status(video_filename)
    if not status:
        raise HTTPException(status_code=404, detail="No processing record found.")
    return _status_to_response(status)


@router.delete("/status/{video_filename}")
def reset_processing_status(video_filename: str, db: Session = Depends(get_db)):
    repo = EventRepository(db)
    status = repo.get_status(video_filename)
    if not status:
        raise HTTPException(status_code=404, detail="No processing record found.")
    db.delete(status)
    db.commit()
    return {"deleted": video_filename, "message": "Video will be reprocessed on next pipeline run."}


# ── Keyframe + Crop serving ───────────────────────────────────────────────────

@router.get("/keyframe")
def get_keyframe(path: str = Query(...)):
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Keyframe not found")
    return FileResponse(path, media_type="image/jpeg")


@router.get("/crop")
def get_crop(path: str = Query(...)):
    """Serve a saved object crop image."""
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Crop not found")
    return FileResponse(path, media_type="image/jpeg")


@router.post("/extract-attributes/{video_filename}")
def extract_attributes(
    video_filename: str,
    db: Session = Depends(get_db),
):
    """
    Phase 6B: Run attribute extraction on a video's best crops.
    Calls minicpm-v on the best crop per tracked object.
    Always re-runs (useful for development / re-processing).
    """
    from app.detection.attribute_processor import AttributeProcessor
    from app.storage.repository import EventRepository

    repo = EventRepository(db)
    if not repo.has_detection_data(video_filename):
        raise HTTPException(
            status_code=404,
            detail="No detection data found. Run Phase 6A pipeline first.",
        )

    try:
        processor = AttributeProcessor(db)
        attributed = processor.run(video_filename)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return {
        "video_filename": video_filename,
        "tracks_attributed": attributed,
        "message": f"Attribute extraction complete. {attributed} track(s) processed.",
    }


@router.get("/attributes/{video_filename}")
def get_track_attributes(
    video_filename: str,
    object_class: Optional[str] = Query(None),
    db: Session = Depends(get_db),
):
    """
    Return all track events with their Phase 6B attributes for a video.
    Only returns entry events (one per track) with attributes populated.
    """
    from app.storage.models import TrackEvent

    q = (
        db.query(TrackEvent)
        .filter(
            TrackEvent.video_filename == video_filename,
            TrackEvent.event_type == "entry",
            TrackEvent.attributes.isnot(None),
        )
    )
    if object_class:
        q = q.filter(TrackEvent.object_class == object_class)

    events = q.order_by(TrackEvent.first_seen_second).all()

    return {
        "video_filename": video_filename,
        "count": len(events),
        "tracks": [
            {
                "track_id": e.track_id,
                "object_class": e.object_class,
                "first_seen": e.first_seen_second,
                "last_seen": e.last_seen_second,
                "duration": e.duration_seconds,
                "best_confidence": round(e.best_confidence or 0, 3),
                "best_crop_path": e.best_crop_path,
                "attributes": e.attributes,
            }
            for e in events
        ],
    }


@router.get("/temporal/{video_filename}")
def get_temporal_analysis(
    video_filename: str,
    object_class: str = Query(None, description="Filter by class: person, car, etc."),
    db: Session = Depends(get_db),
):
    """
    Return temporal behaviour analysis for all tracks in a video.
    Data is read from TrackEvent.attributes["temporal"] — written by TemporalAnalyzer
    during the pipeline run. If temporal data is missing (old video), returns empty list
    with a note to re-run the pipeline.
    """
    from app.storage.models import TrackEvent

    q = (
        db.query(TrackEvent)
        .filter(
            TrackEvent.video_filename == video_filename,
            TrackEvent.event_type == "entry",   # one row per physical track
        )
    )
    if object_class:
        q = q.filter(TrackEvent.object_class == object_class)
    events = q.order_by(TrackEvent.first_seen_second).all()

    results = []
    has_temporal = False
    for ev in events:
        attrs = ev.attributes or {}
        temporal = attrs.get("temporal")
        if temporal:
            has_temporal = True
        row = {
            "track_id": ev.track_id,
            "object_class": ev.object_class,
            "first_seen": ev.first_seen_second,
            "last_seen": ev.last_seen_second,
            "duration_seconds": ev.duration_seconds,
            "best_confidence": ev.best_confidence,
            "best_crop_path": ev.best_crop_path,
            "attributes": attrs,
            "temporal": temporal,
        }
        results.append(row)

    return {
        "video_filename": video_filename,
        "count": len(results),
        "has_temporal_data": has_temporal,
        "tracks": results,
        "note": (
            None if has_temporal
            else "No temporal analysis data. Re-run the pipeline to generate it."
        ),
    }




# ── VideoTimeline: structured event graph + scene understanding ────────────────

@router.get("/video-timeline/{video_filename}")
def get_video_timeline(
    video_filename: str,
    min_second: Optional[float] = Query(None),
    max_second: Optional[float] = Query(None),
    object_class: Optional[str] = Query(None),
    db: Session = Depends(get_db),
):
    """
    Return the full structured timeline for a video (per-second event graph).
    Much richer than /timeline — includes motion events, behaviour labels,
    and scene-level events (crowd, fight_proxy, congestion, etc).
    """
    from app.storage.models import VideoTimeline

    tl = (
        db.query(VideoTimeline)
        .filter(VideoTimeline.video_filename == video_filename)
        .first()
    )
    if not tl:
        raise HTTPException(
            status_code=404,
            detail="No timeline found. Run pipeline or call POST /build-timeline/{video_filename}.",
        )

    entries = tl.timeline_entries or []

    # Apply filters
    if min_second is not None:
        entries = [e for e in entries if e["second"] >= min_second]
    if max_second is not None:
        entries = [e for e in entries if e["second"] <= max_second]
    if object_class:
        entries = [e for e in entries if e.get("object_class") == object_class
                   or e.get("object_class") == "scene"]

    # Enrich entries: if person_label missing (old stored timelines),
    # look it up from TrackEvent.attributes to keep display consistent.
    track_ids_needing_label = [
        e["track_id"] for e in entries
        if e.get("track_id") is not None and not e.get("person_label")
        and e.get("object_class") == "person"
    ]
    if track_ids_needing_label:
        from app.storage.models import TrackEvent
        label_lookup = {}
        te_rows = (
            db.query(TrackEvent.track_id, TrackEvent.attributes)
            .filter(
                TrackEvent.video_filename == video_filename,
                TrackEvent.event_type == "entry",
                TrackEvent.track_id.in_(track_ids_needing_label),
            )
            .all()
        )
        for row_tid, row_attrs in te_rows:
            if row_attrs and row_attrs.get("person_label"):
                label_lookup[row_tid] = row_attrs["person_label"]
        if label_lookup:
            entries = [
                dict(e, person_label=label_lookup[e["track_id"]])
                if e.get("track_id") in label_lookup and not e.get("person_label")
                else e
                for e in entries
            ]

    return {
        "video_filename": video_filename,
        "total_entries": tl.entry_count,
        "filtered_entries": len(entries),
        "duration_seconds": tl.total_duration_seconds,
        "timeline": entries,
        "scene_events": tl.scene_events or [],
        "created_at": tl.created_at.isoformat() if tl.created_at else None,
    }


@router.post("/build-timeline/{video_filename}")
def build_video_timeline(video_filename: str, db: Session = Depends(get_db)):
    """
    Manually trigger timeline rebuild for a video.
    Use this for videos processed before timeline storage was added.
    """
    from app.detection.timeline_builder import TimelineBuilder
    from app.storage.repository import EventRepository

    repo = EventRepository(db)
    if not repo.has_track_event_data(video_filename):
        raise HTTPException(
            status_code=404,
            detail="No track event data found. Run Phase 6A pipeline first.",
        )

    from app.core.config import get_settings
    camera_id = get_settings().camera_id

    try:
        tl = TimelineBuilder(db).build(video_filename, camera_id)
        if not tl:
            raise HTTPException(status_code=500, detail="Timeline build returned empty result.")
        return {
            "video_filename": video_filename,
            "entries": tl.entry_count,
            "scene_events": len(tl.scene_events or []),
            "message": "Timeline built successfully.",
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/run-temporal/{video_filename}")
def run_temporal_analysis(video_filename: str, db: Session = Depends(get_db)):
    """
    Re-run temporal behaviour analysis on an existing video.
    Use for videos processed before TemporalAnalyzer was added (shows 'no data').
    Reads existing TrackEvent + DetectedObject rows — does NOT re-run YOLO.
    """
    from app.storage.models import TrackEvent as TE, DetectedObject as DO
    from app.detection.temporal_analyzer import TemporalAnalyzer
    from app.detection.event_generator import TrackState
    from app.storage.repository import EventRepository
    from sqlalchemy.orm.attributes import flag_modified

    repo = EventRepository(db)
    if not repo.has_track_event_data(video_filename):
        raise HTTPException(status_code=404, detail="No track event data found.")

    # Load DetectedObjects for quadrant data + to rebuild all_seconds per track
    all_dets = (
        db.query(DO)
        .filter(DO.video_filename == video_filename)
        .order_by(DO.frame_second_offset)
        .all()
    )
    det_by_track = {}
    seconds_by_track = {}
    for d in all_dets:
        if d.track_id is not None:
            det_by_track.setdefault(d.track_id, []).append(d)
            seconds_by_track.setdefault(d.track_id, []).append(d.frame_second_offset)

    # Reconstruct TrackState from existing TrackEvent rows.
    # Populate all_seconds from DetectedObject rows for accurate appearance counting.
    entry_events = (
        db.query(TE)
        .filter(TE.video_filename == video_filename, TE.event_type == "entry")
        .all()
    )

    track_states = {}
    for ev in entry_events:
        secs = seconds_by_track.get(ev.track_id) or [ev.first_seen_second, ev.last_seen_second]
        ts = TrackState(
            track_id=ev.track_id,
            object_class=ev.object_class,
            first_seen=ev.first_seen_second,
            last_seen=ev.last_seen_second,
            frame_count=max(2, len(secs)),
            best_second=ev.best_frame_second or ev.first_seen_second,
            best_confidence=ev.best_confidence or 0.5,
            best_crop_path=ev.best_crop_path,
            all_seconds=secs,
        )
        track_states[ev.track_id] = ts

    # Run TemporalAnalyzer
    analyser = TemporalAnalyzer()
    behaviours = analyser.analyze(track_states, det_by_track)
    beh_map = {b.track_id: b for b in behaviours}

    # Write results back into all TrackEvent rows for this video.
    # flag_modified is required so SQLAlchemy detects the JSONB mutation.
    all_events = db.query(TE).filter(TE.video_filename == video_filename).all()
    updated = 0
    for ev in all_events:
        beh = beh_map.get(ev.track_id)
        if beh:
            attrs = dict(ev.attributes or {})
            attrs["temporal"] = beh.to_dict()
            ev.attributes = attrs
            flag_modified(ev, "attributes")
            updated += 1
    db.commit()

    return {
        "video_filename": video_filename,
        "tracks_analysed": len(behaviours),
        "events_updated": updated,
        "behaviours": {b.track_id: b.behaviour for b in behaviours},
        "message": "Temporal analysis complete. Refresh the Temporal tab.",
    }



# ── Semantic Memory Graph ──────────────────────────────────────────────────────

@router.get("/memory-graph/{video_filename}")
def get_memory_graph(
    video_filename: str,
    node_type: Optional[str] = Query(None, description="Filter: identity, behaviour, motion, scene, relationship, timeline"),
    db: Session = Depends(get_db),
):
    """Return all semantic memory graph nodes for a video."""
    from app.storage.models import SemanticMemoryGraph

    q = db.query(SemanticMemoryGraph).filter(
        SemanticMemoryGraph.video_filename == video_filename)
    if node_type:
        q = q.filter(SemanticMemoryGraph.node_type == node_type)
    nodes = q.order_by(SemanticMemoryGraph.start_second).all()

    return {
        "video_filename": video_filename,
        "count": len(nodes),
        "nodes": [
            {
                "id": str(n.id),
                "node_type": n.node_type,
                "node_label": n.node_label,
                "semantic_text": n.semantic_text,
                "track_id": n.track_id,
                "start_second": n.start_second,
                "end_second": n.end_second,
                "confidence": n.confidence,
                "metadata": n.node_meta,
            }
            for n in nodes
        ],
    }


@router.post("/build-memory-graph/{video_filename}")
def build_memory_graph(video_filename: str, db: Session = Depends(get_db)):
    """
    Manually build the Semantic Memory Graph for a video.
    Use for videos processed before memory graph was added.
    First builds timeline if missing, then builds graph.
    """
    from app.storage.memory_graph import MemoryGraphBuilder
    from app.detection.timeline_builder import TimelineBuilder
    from app.storage.repository import EventRepository
    from app.core.config import get_settings

    repo = EventRepository(db)
    if not repo.has_track_event_data(video_filename):
        raise HTTPException(status_code=404, detail="No track event data found.")

    camera_id = get_settings().camera_id

    # Ensure timeline exists first (memory graph depends on it for scene events)
    from app.storage.models import VideoTimeline
    if not db.query(VideoTimeline).filter(
            VideoTimeline.video_filename == video_filename).first():
        try:
            TimelineBuilder(db).build(video_filename, camera_id)
        except Exception as e:
            logger.warning("timeline_build_failed_during_memory", error=str(e))

    try:
        nodes = MemoryGraphBuilder(db).build(video_filename, camera_id)
        if not nodes:
            raise HTTPException(status_code=500, detail="No nodes produced.")
        from collections import Counter
        type_counts = Counter(n.node_type for n in nodes)
        return {
            "video_filename": video_filename,
            "total_nodes": len(nodes),
            "by_type": dict(type_counts),
            "message": "Memory graph built successfully.",
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/index")
def trigger_reindex(db: Session = Depends(get_db)):
    count = CaptionIndexer(db).index_all_unindexed()
    return {"indexed": count}

@router.get("/activity-snapshots/{video_filename}")
def get_activity_snapshots(
    video_filename: str,
    person_label: Optional[str] = Query(None),
    min_duration: float = Query(60.0, description="Min sustained seconds before showing"),
    db: Session = Depends(get_db),
):
    """
    Return validated sustained activity events for a video window.
    Only shows activities that lasted >= min_duration seconds (default 60s).
    Each snapshot was written every 60s by LiveStreamProcessor._maybe_write_activity_snapshot
    when a person had phone/laptop/screen detected for that full minute.

    Answers queries like: "who was using a laptop between 10:00 and 11:00?"
    """
    from app.storage.models import TrackEvent

    snaps = (
        db.query(TrackEvent)
        .filter(
            TrackEvent.video_filename == video_filename,
            TrackEvent.event_type == "activity_snapshot",
        )
        .order_by(TrackEvent.first_seen_second)
        .all()
    )

    if person_label:
        snaps = [s for s in snaps if
                 (s.attributes or {}).get("person_label") == person_label]

    results = []
    for s in snaps:
        attrs = s.attributes or {}
        dur_so_far = attrs.get("duration_so_far", 0)
        if dur_so_far < min_duration:
            continue
        results.append({
            "track_id":      s.track_id,
            "person_label":  attrs.get("person_label", f"track#{s.track_id}"),
            "activity":      attrs.get("activity", "unknown"),
            "objects_nearby": attrs.get("objects_nearby", []),
            "snapshot_time": attrs.get("snapshot_time"),
            "duration_so_far": dur_so_far,
            "best_crop_path": s.best_crop_path,
            "window":        video_filename,
        })

    return {
        "video_filename": video_filename,
        "count": len(results),
        "min_duration_filter": min_duration,
        "snapshots": results,
    }


@router.get("/objects-in-frame/{video_filename}")
def get_objects_in_frame(
    video_filename: str,
    track_id: Optional[int] = Query(None),
    db: Session = Depends(get_db),
):
    """
    Return objects detected near each person in a video window.
    Reads objects_nearby from TrackEvent.attributes — populated live by
    ActivityDetector.detect_on_context() every 5 frames.
    Groups by track_id showing what objects were seen near each person.
    Used in Detections tab and by factual Ask queries.
    """
    from app.storage.models import TrackEvent

    q = db.query(TrackEvent).filter(
        TrackEvent.video_filename == video_filename,
        TrackEvent.event_type == "entry",
    )
    if track_id is not None:
        q = q.filter(TrackEvent.track_id == track_id)
    events = q.order_by(TrackEvent.first_seen_second).all()

    results = []
    for ev in events:
        attrs = ev.attributes or {}
        objs = attrs.get("objects_nearby") or []
        activity = attrs.get("activity_hint") or attrs.get("activity_caption") or "present"
        results.append({
            "track_id":      ev.track_id,
            "person_label":  attrs.get("person_label", f"track#{ev.track_id}"),
            "objects_nearby": objs,
            "activity":      activity,
            "activity_caption": attrs.get("activity_caption"),
            "best_crop_path": ev.best_crop_path,
            "first_seen":    ev.first_seen_second,
            "last_seen":     ev.last_seen_second,
            "duration":      ev.duration_seconds,
        })

    return {
        "video_filename": video_filename,
        "count": len(results),
        "tracks": results,
    }