from typing import Optional
from datetime import datetime
from sqlalchemy.orm import Session
from app.storage.models import (
    Caption, Event, VideoSummary, ProcessingStatus,
    DetectedObject, TrackEvent,
)


class EventRepository:
    def __init__(self, db: Session):
        self.db = db

    # ── Legacy caption methods (unchanged) ────────────────────────────────────

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
        row = self.get_status(video_filename)
        if row is None:
            row = ProcessingStatus(video_filename=video_filename, camera_id=camera_id)
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
        return self.upsert_status(video_filename, camera_id, status="skipped")

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

    # ── Phase 6A: DetectedObject persistence ─────────────────────────────────

    def save_detected_object(
        self,
        video_filename: str,
        camera_id: str,
        frame_second_offset: float,
        object_class: str,
        confidence: float,
        bbox_x1: float, bbox_y1: float, bbox_x2: float, bbox_y2: float,
        track_id: Optional[int],
        frame_quadrant: str,
        crop_path: Optional[str],
        rag_text: str,
    ) -> DetectedObject:
        obj = DetectedObject(
            video_filename=video_filename,
            camera_id=camera_id,
            frame_second_offset=frame_second_offset,
            object_class=object_class,
            confidence=confidence,
            bbox_x1=bbox_x1, bbox_y1=bbox_y1,
            bbox_x2=bbox_x2, bbox_y2=bbox_y2,
            track_id=track_id,
            frame_quadrant=frame_quadrant,
            crop_path=crop_path,
            rag_text=rag_text,
        )
        self.db.add(obj)
        self.db.commit()
        self.db.refresh(obj)
        return obj

    def get_detected_objects(
        self,
        video_filename: str,
        object_class: Optional[str] = None,
        track_id: Optional[int] = None,
        min_second: Optional[float] = None,
        max_second: Optional[float] = None,
    ) -> list[DetectedObject]:
        q = (
            self.db.query(DetectedObject)
            .filter(DetectedObject.video_filename == video_filename)
        )
        if object_class:
            q = q.filter(DetectedObject.object_class == object_class)
        if track_id is not None:
            q = q.filter(DetectedObject.track_id == track_id)
        if min_second is not None:
            q = q.filter(DetectedObject.frame_second_offset >= min_second)
        if max_second is not None:
            q = q.filter(DetectedObject.frame_second_offset <= max_second)
        return q.order_by(DetectedObject.frame_second_offset).all()

    def count_detected_objects(self, video_filename: str) -> int:
        return (
            self.db.query(DetectedObject)
            .filter(DetectedObject.video_filename == video_filename)
            .count()
        )

    # ── Phase 6A: TrackEvent persistence ─────────────────────────────────────

    def save_track_event(
        self,
        video_filename: str,
        camera_id: str,
        track_id: int,
        object_class: str,
        event_type: str,
        first_seen_second: float,
        last_seen_second: float,
        duration_seconds: float,
        best_frame_second: Optional[float],
        best_crop_path: Optional[str],
        best_confidence: Optional[float],
        rag_text: str,
        attributes: Optional[dict] = None,
    ) -> TrackEvent:
        event = TrackEvent(
            video_filename=video_filename,
            camera_id=camera_id,
            track_id=track_id,
            object_class=object_class,
            event_type=event_type,
            first_seen_second=first_seen_second,
            last_seen_second=last_seen_second,
            duration_seconds=duration_seconds,
            best_frame_second=best_frame_second,
            best_crop_path=best_crop_path,
            best_confidence=best_confidence,
            rag_text=rag_text,
            attributes=attributes,
        )
        self.db.add(event)
        self.db.commit()
        self.db.refresh(event)
        return event

    def get_track_events(
        self,
        video_filename: str,
        event_type: Optional[str] = None,
        object_class: Optional[str] = None,
        track_id: Optional[int] = None,
    ) -> list[TrackEvent]:
        q = (
            self.db.query(TrackEvent)
            .filter(TrackEvent.video_filename == video_filename)
        )
        if event_type:
            q = q.filter(TrackEvent.event_type == event_type)
        if object_class:
            q = q.filter(TrackEvent.object_class == object_class)
        if track_id is not None:
            q = q.filter(TrackEvent.track_id == track_id)
        return q.order_by(TrackEvent.first_seen_second).all()

    def get_track_summary(self, video_filename: str) -> dict:
        """
        Return a structured summary of all tracks in a video.
        Used by the summary generator and API.
        """
        events = self.get_track_events(video_filename)
        by_class: dict[str, list] = {}
        for ev in events:
            if ev.event_type == "entry":
                by_class.setdefault(ev.object_class, []).append({
                    "track_id": ev.track_id,
                    "first_seen": ev.first_seen_second,
                    "last_seen": ev.last_seen_second,
                    "duration": ev.duration_seconds,
                })
        return by_class

    def mark_6b_completed(self, video_filename: str, tracks_attributed: int) -> None:
        """Mark Phase 6B attribute extraction as done for this video."""
        row = self.get_status(video_filename)
        if row:
            row.phase_6b_completed = True
            row.phase_6b_tracks_attributed = tracks_attributed
            row.updated_at = __import__("datetime").datetime.utcnow()
            self.db.commit()

    def has_6b_completed(self, video_filename: str) -> bool:
        """Check if Phase 6B attribute extraction has already run for this video."""
        row = self.get_status(video_filename)
        return row is not None and bool(row.phase_6b_completed)

    def has_detection_data(self, video_filename: str) -> bool:
        """Check if Phase 6A detection data exists for this video."""
        return (
            self.db.query(DetectedObject)
            .filter(DetectedObject.video_filename == video_filename)
            .first()
        ) is not None