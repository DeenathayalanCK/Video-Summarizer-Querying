from datetime import datetime
from sqlalchemy.orm import Session

from app.storage.database import SessionLocal
from app.storage.repository import EventRepository
from app.core.event_bus import event_bus
from app.core.logging import get_logger


logger = get_logger()


def handle_generic_event(payload: dict):
    """
    Scene-agnostic event handler. Reads event_type from payload — no hardcoding.
    Expected payload keys:
        camera_id (str), event_type (str), frame_timestamp (datetime),
        track_id (int, optional), zone (str, optional),
        confidence (float, optional), metadata (dict, optional)
    """
    db: Session = SessionLocal()
    try:
        repo = EventRepository(db)

        event = repo.save_event(
            camera_id=payload["camera_id"],
            event_type=payload["event_type"],       # driven by caller, not hardcoded
            track_id=payload.get("track_id"),
            zone=payload.get("zone"),
            confidence=payload.get("confidence"),
            frame_timestamp=payload["frame_timestamp"],
            event_timestamp=datetime.utcnow(),
            metadata=payload.get("metadata"),
        )

        logger.info(
            "event_persisted",
            event_id=str(event.id),
            camera_id=event.camera_id,
            event_type=event.event_type,
        )

    finally:
        db.close()


# Wire handler to the event bus for any event type published as "video_event"
event_bus.subscribe("video_event", handle_generic_event)
