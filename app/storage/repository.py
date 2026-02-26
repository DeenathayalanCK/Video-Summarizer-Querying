from typing import Optional
from sqlalchemy.orm import Session
from sqlalchemy import func
from app.storage.models import Caption, Event, VideoSummary
from datetime import datetime


class EventRepository:
    def __init__(self, db: Session):
        self.db = db

    def save_event(
        self,
        camera_id: str,
        event_type: str,
        frame_timestamp: datetime,
        event_timestamp: datetime,
        track_id: Optional[int] = None,
        zone: Optional[str] = None,
        confidence: Optional[float] = None,
        metadata: Optional[dict] = None,
        schema_version: int = 1,
    ) -> Event:
        event = Event(
            camera_id=camera_id,
            event_type=event_type,
            track_id=track_id,
            zone=zone,
            confidence=confidence,
            frame_timestamp=frame_timestamp,
            event_timestamp=event_timestamp,
            event_metadata=metadata,
            schema_version=schema_version,
        )
        self.db.add(event)
        self.db.commit()
        self.db.refresh(event)
        return event

    def save_caption(
        self,
        camera_id: str,
        video_filename: str,
        frame_second_offset: float,
        absolute_timestamp: Optional[datetime],
        keyframe_path: str,
        caption_text: str,
    ) -> Caption:
        caption = Caption(
            camera_id=camera_id,
            video_filename=video_filename,
            frame_second_offset=frame_second_offset,
            absolute_timestamp=absolute_timestamp,
            keyframe_path=keyframe_path,
            caption_text=caption_text,
        )
        self.db.add(caption)
        self.db.commit()
        self.db.refresh(caption)
        return caption

    def get_summary(
        self, video_filename: str, camera_id: str
    ) -> Optional[VideoSummary]:
        return (
            self.db.query(VideoSummary)
            .filter(
                VideoSummary.video_filename == video_filename,
                VideoSummary.camera_id == camera_id,
            )
            .first()
        )

    def list_summaries(self) -> list[VideoSummary]:
        return (
            self.db.query(VideoSummary)
            .order_by(VideoSummary.video_filename)
            .all()
        )