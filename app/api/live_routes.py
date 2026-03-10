"""
live_routes.py — API endpoints for live RTSP stream.

Endpoints:
  POST /live/start          — start the stream processor
  POST /live/stop           — stop it
  GET  /live/status         — running state + active person count
  GET  /live/stream         — MJPEG stream (img src="...")
  GET  /live/active         — active persons (for sidebar cards)
  GET  /live/sessions       — today's person sessions (for table)
  GET  /live/sessions/all   — all sessions ever
  GET  /live/identities     — all known identities (P1, P2...)
  DELETE /live/identities   — wipe identity DB (reset P1, P2 counters)
  GET  /live/crop/{label}   — latest crop image for a person label
"""

import os
from datetime import datetime, date
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse, FileResponse, Response
from sqlalchemy.orm import Session

from app.storage.database import SessionLocal

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
from app.storage.live_models import PersonSession, PersonIdentity
from app.vision.live_stream_processor import LiveStreamProcessor
from app.core.logging import get_logger

router = APIRouter(prefix="/live", tags=["live"])
_log = get_logger()


# ── Stream control ────────────────────────────────────────────────────────────

@router.post("/start")
def start_stream():
    """Start the RTSP live stream processor."""
    # Ensure WindowManager rotation loop is running so every 5-min window
    # triggers attribute extraction, temporal analysis, and summarization.
    from app.vision.window_manager import WindowManager
    wm = WindowManager.get_instance()
    if not (wm._thread and wm._thread.is_alive()):
        wm.start()
        _log.info("window_manager_started_via_live_start")

    proc = LiveStreamProcessor.get_instance()
    if proc.is_running:
        return {"status": "already_running"}
    ok = proc.start()
    if not ok:
        raise HTTPException(
            status_code=400,
            detail="Failed to start stream. Check VIDEO_INPUT_PATH in .env.")
    return {"status": "started"}


@router.post("/stop")
def stop_stream():
    """Stop the RTSP live stream processor."""
    proc = LiveStreamProcessor.get_instance()
    proc.stop()

    # Stop WindowManager: closes the current window and queues its
    # post-processing (attributes → temporal → summary).
    from app.vision.window_manager import WindowManager
    wm = WindowManager.get_instance()
    if wm._thread and wm._thread.is_alive():
        wm.stop()
    return {"status": "stopped"}


@router.get("/status")
def stream_status(db: Session = Depends(get_db)):
    """Current stream state and stats."""
    proc = LiveStreamProcessor.get_instance()
    active = proc.get_active_tracks()

    # Today's session count
    today_start = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
    today_count = db.query(PersonSession).filter(
        PersonSession.entry_time >= today_start).count()

    total_identities = db.query(PersonIdentity).count()

    return {
        "running":           proc.is_running,
        "active_persons":    len([t for t in active if t["active"]]),
        "total_identities":  total_identities,
        "sessions_today":    today_count,
        "frame_age_seconds": round(proc.frame_buffer.age(), 1),
    }


# ── MJPEG stream ──────────────────────────────────────────────────────────────

@router.get("/stream")
def mjpeg_stream():
    """
    MJPEG stream endpoint. Use as:  <img src="/api/v1/live/stream">
    Streams annotated frames as multipart/x-mixed-replace.
    Falls back to a placeholder frame if stream not running.
    """
    proc = LiveStreamProcessor.get_instance()

    def generate():
        import time
        boundary = b"--frame\r\n"
        header   = b"Content-Type: image/jpeg\r\n\r\n"

        while True:
            frame = proc.get_frame()

            if frame is None:
                # Stream not running — send a placeholder
                frame = _make_placeholder_frame()

            yield boundary + header + frame + b"\r\n"
            time.sleep(0.08)   # ~12fps max for browser

    return StreamingResponse(
        generate(),
        media_type="multipart/x-mixed-replace; boundary=frame",
        headers={"Cache-Control": "no-cache, no-store"},
    )


# ── Person data ───────────────────────────────────────────────────────────────

@router.get("/active")
def active_persons():
    """
    Currently active (in-frame) persons.
    Used by the Live tab sidebar cards. Updates every poll.
    """
    proc = LiveStreamProcessor.get_instance()
    tracks = proc.get_active_tracks()
    # Only return those still within timeout window
    from app.core.config import get_settings
    timeout = get_settings().live_exit_timeout_seconds
    active = [t for t in tracks if t["active"]]
    return {"persons": active, "count": len(active)}


@router.get("/activity-log")
def live_activity_log():
    """
    Current activity of all active persons — phone, laptop, screen detections.
    Only returns persons with a non-trivial activity_hint.
    Updated every 5 frames (~2.5s at 2 FPS) by ActivityDetector.detect_on_context().
    """
    proc = LiveStreamProcessor.get_instance()
    tracks = proc.get_active_tracks()
    log = []
    for t in tracks:
        if not t.get("active"):
            continue
        hint = t.get("activity_hint", "present")
        objs = t.get("objects_nearby", [])
        if hint and hint != "present":
            log.append({
                "person_label":  t["person_label"],
                "activity":      hint,
                "objects_nearby": objs,
                "duration_seconds": t.get("duration_seconds", 0),
                "best_crop_path": t.get("best_crop_path"),
                "last_seen":     t.get("last_seen"),
            })
    return {"activity_log": log, "count": len(log)}


@router.get("/sessions")
def sessions_today(db: Session = Depends(get_db)):
    """All sessions from today, newest first."""
    today_start = datetime.utcnow().replace(
        hour=0, minute=0, second=0, microsecond=0)
    rows = (
        db.query(PersonSession)
        .filter(PersonSession.entry_time >= today_start)
        .order_by(PersonSession.entry_time.desc())
        .limit(200)
        .all()
    )
    return {"sessions": [_session_to_dict(r) for r in rows]}


@router.get("/sessions/all")
def all_sessions(
    limit: int = 500,
    person_label: Optional[str] = None,
    db: Session = Depends(get_db),
):
    """All sessions, newest first. Optionally filter by person label."""
    q = db.query(PersonSession).order_by(PersonSession.entry_time.desc())
    if person_label:
        q = q.filter(PersonSession.person_label == person_label)
    rows = q.limit(limit).all()
    return {"sessions": [_session_to_dict(r) for r in rows]}


@router.get("/identities")
def list_identities(db: Session = Depends(get_db)):
    """All known person identities."""
    rows = db.query(PersonIdentity).order_by(PersonIdentity.person_label).all()
    return {
        "identities": [
            {
                "person_label":  r.person_label,
                "total_visits":  r.total_visits,
                "first_seen_at": r.first_seen_at.isoformat() if r.first_seen_at else None,
                "last_seen_at":  r.last_seen_at.isoformat() if r.last_seen_at else None,
                "best_crop_path": r.best_crop_path,
                "attributes":    r.attributes,
            }
            for r in rows
        ]
    }


@router.delete("/identities")
def reset_identities(db: Session = Depends(get_db)):
    """
    Wipe all person identities and sessions.
    Resets P1/P2 counters. Use when starting fresh.
    """
    proc = LiveStreamProcessor.get_instance()
    if proc.is_running:
        proc.stop()

    db.query(PersonSession).delete()
    db.query(PersonIdentity).delete()
    db.commit()

    # Reset label counter
    proc._next_label_num = 1
    _log.info("live_identities_reset")
    return {"status": "reset", "message": "All identities and sessions cleared"}


# ── Crop images ───────────────────────────────────────────────────────────────

@router.get("/crop/{person_label}")
def get_crop(person_label: str, db: Session = Depends(get_db)):
    """
    Return the best crop image for a person label.
    Falls back to newest file in live_crops matching the label if DB path is stale
    (happens when the container restarted and the named volume was recreated).
    """
    from app.core.config import get_settings as _gs
    identity = db.query(PersonIdentity).filter(
        PersonIdentity.person_label == person_label).first()

    # Try DB path first
    crop_path = identity.best_crop_path if identity else None
    if crop_path and os.path.exists(crop_path):
        return FileResponse(crop_path, media_type="image/jpeg",
                            headers={"Cache-Control": "max-age=30"})

    # Fallback: scan live_crops dir for most recent file matching label prefix
    try:
        crops_dir = _gs().live_crops_path
        prefix    = person_label + "_"
        matches   = [
            os.path.join(crops_dir, f)
            for f in os.listdir(crops_dir)
            if f.startswith(prefix) and f.endswith(".jpg") and "_face" not in f
        ]
        if matches:
            newest = max(matches, key=os.path.getmtime)
            # Update DB so next request hits cache
            if identity:
                identity.best_crop_path = newest
                try: db.commit()
                except Exception: db.rollback()
            return FileResponse(newest, media_type="image/jpeg",
                                headers={"Cache-Control": "max-age=10"})
    except Exception:
        pass

    raise HTTPException(status_code=404, detail="No crop available for " + person_label)


# ── Person history detail ────────────────────────────────────────────────────


@router.get("/face/{person_label}")
def get_live_face(person_label: str, db: Session = Depends(get_db)):
    """
    Return the latest face crop for a live person identity.
    Reads from the most recent entry TrackEvent where face_detected=true.
    Used by the live person cards in the UI to show face thumbnail.
    """
    from app.storage.models import TrackEvent
    from app.storage.live_models import PersonIdentity

    identity = db.query(PersonIdentity).filter(
        PersonIdentity.person_label == person_label
    ).first()

    if not identity:
        return {"person_label": person_label, "face_detected": False,
                "error": "identity_not_found"}

    # Find most recent entry event with face_detected=true
    # We query all entry events and filter Python-side to avoid JSONB operator issues
    recent_events = (
        db.query(TrackEvent)
        .filter(
            TrackEvent.event_type == "entry",
        )
        .order_by(TrackEvent.first_seen_second.desc())
        .limit(200)
        .all()
    )

    for ev in recent_events:
        attrs = ev.attributes or {}
        if attrs.get("person_label") == person_label and attrs.get("face_detected"):
            return {
                "person_label":    person_label,
                "face_detected":   True,
                "face_crop_path":  attrs.get("face_crop_path"),
                "face_confidence": attrs.get("face_confidence", 0.0),
                "face_method":     attrs.get("face_method"),
                "window":          ev.video_filename,
                "best_crop_path":  ev.best_crop_path,
            }

    return {
        "person_label": person_label,
        "face_detected": False,
        "best_crop_path": identity.best_crop_path,
    }


@router.get("/person/{person_label}")
def person_detail(person_label: str, db: Session = Depends(get_db)):
    """
    Full history for one person: identity + all sessions + crop.
    Used by the Live History tab person detail view.
    """
    identity = db.query(PersonIdentity).filter(
        PersonIdentity.person_label == person_label).first()
    if not identity:
        raise HTTPException(status_code=404, detail="Person not found")

    sessions = (
        db.query(PersonSession)
        .filter(PersonSession.person_label == person_label)
        .order_by(PersonSession.entry_time.desc())
        .limit(50)
        .all()
    )

    # Compute stats
    completed = [s for s in sessions if s.duration_seconds and s.duration_seconds > 0]
    avg_dur   = (sum(s.duration_seconds for s in completed) / len(completed)
                 if completed else 0)
    states    = [s.last_state for s in sessions if s.last_state]
    from collections import Counter
    state_counts = dict(Counter(states).most_common(5))

    # Activity snapshots for this person across all windows
    from app.storage.models import TrackEvent
    person_label_val = identity.person_label
    snap_rows = (
        db.query(TrackEvent)
        .filter(
            TrackEvent.event_type == "activity_snapshot",
            TrackEvent.attributes[("person_label")].astext == person_label_val,
        )
        .order_by(TrackEvent.first_seen_second.desc())
        .limit(50)
        .all()
    )
    activity_snaps = [
        {
            "activity":      (s.attributes or {}).get("activity"),
            "objects_nearby": (s.attributes or {}).get("objects_nearby", []),
            "snapshot_time": (s.attributes or {}).get("snapshot_time"),
            "duration_so_far": (s.attributes or {}).get("duration_so_far", 0),
            "window":        s.video_filename,
        }
        for s in snap_rows
        if (s.attributes or {}).get("duration_so_far", 0) >= 60
    ]

    return {
        "person_label":    identity.person_label,
        "total_visits":    identity.total_visits,
        "first_seen_at":   identity.first_seen_at.isoformat() if identity.first_seen_at else None,
        "last_seen_at":    identity.last_seen_at.isoformat() if identity.last_seen_at else None,
        "best_crop_path":  identity.best_crop_path,
        "attributes":      identity.attributes or {},
        "avg_duration_seconds": round(avg_dur, 1),
        "state_breakdown": state_counts,
        "sessions":        [_session_to_dict(s) for s in sessions],
        "activity_snapshots": activity_snaps,
    }


@router.get("/history")
def live_history(db: Session = Depends(get_db)):
    """
    Full live history: all known identities with their session stats.
    Used by the Live History tab main view.
    """
    identities = (
        db.query(PersonIdentity)
        .order_by(PersonIdentity.last_seen_at.desc())
        .all()
    )

    result = []
    for ident in identities:
        sessions = (
            db.query(PersonSession)
            .filter(PersonSession.person_label == ident.person_label)
            .order_by(PersonSession.entry_time.desc())
            .all()
        )
        completed    = [s for s in sessions if s.duration_seconds and s.duration_seconds > 0]
        avg_dur      = (sum(s.duration_seconds for s in completed) / len(completed)
                        if completed else 0)
        states       = [s.last_state for s in sessions if s.last_state]
        from collections import Counter
        top_state    = Counter(states).most_common(1)[0][0] if states else "unknown"
        active_sess  = next((s for s in sessions if s.is_active), None)

        result.append({
            "person_label":    ident.person_label,
            "total_visits":    ident.total_visits,
            "first_seen_at":   ident.first_seen_at.isoformat() if ident.first_seen_at else None,
            "last_seen_at":    ident.last_seen_at.isoformat() if ident.last_seen_at else None,
            "best_crop_path":  ident.best_crop_path,
            "attributes":      ident.attributes or {},
            "avg_duration_seconds": round(avg_dur, 1),
            "top_state":       top_state,
            "is_active_now":   active_sess is not None,
            "session_count":   len(sessions),
            "last_session":    _session_to_dict(sessions[0]) if sessions else None,
        })

    return {"persons": result, "total": len(result)}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _session_to_dict(s: PersonSession) -> dict:
    return {
        "id":               str(s.id),
        "person_label":     s.person_label,
        "track_id":         s.track_id,
        "entry_time":       s.entry_time.isoformat() if s.entry_time else None,
        "exit_time":        s.exit_time.isoformat() if s.exit_time else None,
        "duration_seconds": round(s.duration_seconds or 0, 1),
        "last_state":       s.last_state,
        "is_active":        s.is_active,
        "best_crop_path":   s.best_crop_path,
        "best_confidence":  s.best_confidence,
    }


def _make_placeholder_frame() -> bytes:
    """Return a minimal JPEG placeholder when stream is offline."""
    import cv2
    import numpy as np
    img = np.zeros((360, 640, 3), dtype=np.uint8)
    cv2.putText(img, "Stream offline", (180, 170),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (80, 80, 80), 2)
    cv2.putText(img, "POST /api/v1/live/start to begin", (100, 220),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (60, 60, 60), 1)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 60])
    return bytes(buf) if ok else b""