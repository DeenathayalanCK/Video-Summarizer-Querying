from datetime import datetime
from sqlalchemy.orm import Session

from app.storage.database import SessionLocal
from app.storage.repository import EventRepository
from app.core.logging import get_logger


logger = get_logger()


def handle_vehicle_entry(payload: dict):
    db: Session = SessionLocal()
    try:
        repo = EventRepository(db)

        event = repo.save_event(
            camera_id=payload["camera_id"],
            event_type="vehicle_entry",
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
        )

    finally:
        db.close()