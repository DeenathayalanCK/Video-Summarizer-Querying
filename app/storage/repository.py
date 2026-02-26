from typing import Optional
from datetime import datetime
from sqlalchemy.orm import Session
from app.storage.models import Caption, Event, VideoSummary, ProcessingStatus


class EventRepository:
    def __init__(self, db: Session):
        self.db = db

    def save_event(self, camera_id, event_type, frame_timestamp,
                   event_timestamp, track_id=None, zone=None,
                   confidence=None, metadata=None, schema_version=1) -> Event:
        event = Event(
            camera_id=camera_id, event_type=event_type, track_id=track_id,
            zone=zone, confidence=confidence, frame_timestamp=frame_timestamp,
            event_timestamp=event_timestamp, event_metadata=metadata,
            schema_version=schema_version,
        )
        self.db.add(event)
        self.db.commit()
        self.db.refresh(event)
        return event

    def save_caption(self, camera_id, video_filename, frame_second_offset,
                     absolute_timestamp, keyframe_path, caption_text) -> Caption:
        caption = Caption(
            camera_id=camera_id, video_filename=video_filename,
            frame_second_offset=frame_second_offset,
            absolute_timestamp=absolute_timestamp,
            keyframe_path=keyframe_path, caption_text=caption_text,
        )
        self.db.add(caption)
        self.db.commit()
        self.db.refresh(caption)
        return caption

    # ── ProcessingStatus ──────────────────────────────────────────────────────

    def get_status(self, video_filename: str) -> Optional[ProcessingStatus]:
        return (
            self.db.query(ProcessingStatus)
            .filter(ProcessingStatus.video_filename == video_filename)
            .first()
        )

    def list_statuses(self) -> list[ProcessingStatus]:
        return (
            self.db.query(ProcessingStatus)
            .order_by(ProcessingStatus.created_at)
            .all()
        )

    def upsert_status(self, video_filename: str, camera_id: str, **kwargs) -> ProcessingStatus:
        """Create or update a ProcessingStatus row."""
        row = self.get_status(video_filename)
        if row is None:
            row = ProcessingStatus(
                video_filename=video_filename,
                camera_id=camera_id,
            )
            self.db.add(row)
        for key, value in kwargs.items():
            setattr(row, key, value)
        row.updated_at = datetime.utcnow()
        self.db.commit()
        self.db.refresh(row)
        return row

    def mark_running(self, video_filename: str, camera_id: str,
                     total_frames: Optional[int] = None) -> ProcessingStatus:
        return self.upsert_status(
            video_filename, camera_id,
            status="running",
            started_at=datetime.utcnow(),
            total_frames_sampled=total_frames,
            scenes_detected=0,
            scenes_captioned=0,
            last_error=None,
        )

    def mark_scene_detected(self, video_filename: str, second: float) -> None:
        row = self.get_status(video_filename)
        if row:
            row.scenes_detected = (row.scenes_detected or 0) + 1
            row.current_second = second
            row.updated_at = datetime.utcnow()
            self.db.commit()

    def mark_scene_captioned(self, video_filename: str) -> None:
        row = self.get_status(video_filename)
        if row:
            row.scenes_captioned = (row.scenes_captioned or 0) + 1
            row.updated_at = datetime.utcnow()
            self.db.commit()

    def mark_completed(self, video_filename: str) -> ProcessingStatus:
        return self.upsert_status(
            video_filename, "",
            status="completed",
            completed_at=datetime.utcnow(),
        )

    def mark_failed(self, video_filename: str, error: str) -> ProcessingStatus:
        row = self.get_status(video_filename)
        camera_id = row.camera_id if row else ""
        return self.upsert_status(
            video_filename, camera_id,
            status="failed",
            last_error=error,
            error_count=(row.error_count + 1) if row else 1,
        )

    def mark_skipped(self, video_filename: str, camera_id: str) -> ProcessingStatus:
        return self.upsert_status(
            video_filename, camera_id,
            status="skipped",
        )

    def is_completed(self, video_filename: str) -> bool:
        row = self.get_status(video_filename)
        return row is not None and row.status == "completed"

    # ── Summaries ─────────────────────────────────────────────────────────────

    def get_summary(self, video_filename, camera_id) -> Optional[VideoSummary]:
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