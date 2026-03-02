import uuid
from datetime import datetime
from sqlalchemy import (
    Column, String, Integer, Float,
    DateTime, Index, Text, ForeignKey, Boolean,
)
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import declarative_base, relationship
from sqlalchemy.sql import func
from pgvector.sqlalchemy import Vector

Base = declarative_base()


# ── Legacy tables (kept intact — caption pipeline still works) ─────────────────

class Event(Base):
    __tablename__ = "events"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    camera_id = Column(String(100), nullable=False, index=True)
    event_type = Column(String(100), nullable=False, index=True)
    track_id = Column(Integer, nullable=True)
    zone = Column(String(100), nullable=True)
    confidence = Column(Float, nullable=True)
    frame_timestamp = Column(DateTime(timezone=True), nullable=False)
    event_timestamp = Column(DateTime(timezone=True), nullable=False)
    event_metadata = Column("metadata", JSONB, nullable=True)
    schema_version = Column(Integer, nullable=False, default=1)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)


class Caption(Base):
    __tablename__ = "captions"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    camera_id = Column(String, nullable=False, index=True)
    video_filename = Column(String, nullable=False, index=True)
    frame_second_offset = Column(Float, nullable=False)
    absolute_timestamp = Column(DateTime, nullable=True)
    caption_text = Column(Text, nullable=False)
    keyframe_path = Column(String, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    embedding = relationship(
        "CaptionEmbedding",
        back_populates="caption",
        uselist=False,
        cascade="all, delete-orphan",
    )


class CaptionEmbedding(Base):
    __tablename__ = "caption_embeddings"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    caption_id = Column(
        UUID(as_uuid=True),
        ForeignKey("captions.id", ondelete="CASCADE"),
        nullable=False, unique=True, index=True,
    )
    embedding = Column(Vector(768), nullable=False)
    model_name = Column(String, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    caption = relationship("Caption", back_populates="embedding")


class VideoSummary(Base):
    __tablename__ = "video_summaries"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    video_filename = Column(String, nullable=False, index=True)
    camera_id = Column(String, nullable=False)
    summary_text = Column(Text, nullable=False)
    caption_count = Column(Integer, nullable=False)
    duration_seconds = Column(Float, nullable=False)
    model_name = Column(String, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class ProcessingStatus(Base):
    """
    Tracks per-video processing state.
    State machine: pending → running → completed | failed | skipped
    """
    __tablename__ = "processing_status"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    video_filename = Column(String, nullable=False, unique=True, index=True)
    camera_id = Column(String, nullable=False)
    status = Column(String(20), nullable=False, default="pending")

    total_frames_sampled = Column(Integer, nullable=True)
    scenes_detected = Column(Integer, nullable=False, default=0)
    scenes_captioned = Column(Integer, nullable=False, default=0)
    current_second = Column(Float, nullable=True)

    last_error = Column(Text, nullable=True)
    error_count = Column(Integer, nullable=False, default=0)

    started_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    created_at = Column(DateTime, default=datetime.utcnow)

    # Phase 6B: track whether attribute extraction has run
    phase_6b_completed = Column(Boolean, nullable=False, default=False)
    phase_6b_tracks_attributed = Column(Integer, nullable=True)


# ── Phase 6A: Detection + Tracking tables ──────────────────────────────────────

class DetectedObject(Base):
    """
    One row per detected object instance per frame.
    Stores the raw YOLO output + ByteTrack track_id.

    A single physical object (e.g. car #7) produces many DetectedObject rows
    across frames — one per frame it appears in. They share the same track_id.
    The TrackEvent table stores the lifecycle events (entry, exit, dwell).
    """
    __tablename__ = "detected_objects"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)

    # Video context
    video_filename = Column(String, nullable=False, index=True)
    camera_id = Column(String, nullable=False, index=True)
    frame_second_offset = Column(Float, nullable=False)

    # Detection
    object_class = Column(String(50), nullable=False, index=True)
    # e.g. "person", "car", "truck", "motorcycle", "bus", "bicycle"
    confidence = Column(Float, nullable=False)

    # Bounding box (normalized 0-1 relative to frame dimensions)
    bbox_x1 = Column(Float, nullable=False)
    bbox_y1 = Column(Float, nullable=False)
    bbox_x2 = Column(Float, nullable=False)
    bbox_y2 = Column(Float, nullable=False)

    # Tracking
    track_id = Column(Integer, nullable=True, index=True)
    # None if tracker lost this object; integer if ByteTrack assigned an ID

    # Frame position helpers (for UI and querying)
    frame_quadrant = Column(String(20), nullable=True)
    # "top-left", "top-right", "bottom-left", "bottom-right", "center"

    # Crop saved to disk for attribute extraction (Phase 6B)
    crop_path = Column(String, nullable=True)

    # Phase 6B attribute fields — null in Phase 6A, filled in 6B
    # Vehicle attributes
    vehicle_color = Column(String(50), nullable=True)
    vehicle_type = Column(String(50), nullable=True)    # sedan, suv, truck, etc.
    vehicle_make = Column(String(100), nullable=True)   # Toyota, Honda, etc.

    # Person attributes
    person_gender = Column(String(20), nullable=True)
    person_clothing_top = Column(String(100), nullable=True)
    person_clothing_bottom = Column(String(100), nullable=True)

    # RAG text — generated from detection data, embedded for search
    rag_text = Column(Text, nullable=True)

    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationship to embeddings
    embedding = relationship(
        "DetectedObjectEmbedding",
        back_populates="detected_object",
        uselist=False,
        cascade="all, delete-orphan",
    )


class DetectedObjectEmbedding(Base):
    """Vector embedding of the DetectedObject's rag_text for semantic search."""
    __tablename__ = "detected_object_embeddings"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    object_id = Column(
        UUID(as_uuid=True),
        ForeignKey("detected_objects.id", ondelete="CASCADE"),
        nullable=False, unique=True, index=True,
    )
    embedding = Column(Vector(768), nullable=False)
    model_name = Column(String, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    detected_object = relationship("DetectedObject", back_populates="embedding")


class TrackEvent(Base):
    """
    Lifecycle events for a tracked object across a video.
    Generated by the EventGenerator after the full frame loop completes.

    One TrackEvent per meaningful state transition:
      entry  — track_id seen for the first time
      exit   — track_id not seen for > exit_threshold seconds
      dwell  — track_id present continuously for > dwell_threshold seconds
    """
    __tablename__ = "track_events"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)

    video_filename = Column(String, nullable=False, index=True)
    camera_id = Column(String, nullable=False)
    track_id = Column(Integer, nullable=False, index=True)
    object_class = Column(String(50), nullable=False)

    event_type = Column(String(20), nullable=False, index=True)
    # "entry" | "exit" | "dwell"

    # Time span
    first_seen_second = Column(Float, nullable=False)
    last_seen_second = Column(Float, nullable=False)
    duration_seconds = Column(Float, nullable=False)

    # Best frame for this track (highest confidence detection)
    best_frame_second = Column(Float, nullable=True)
    best_crop_path = Column(String, nullable=True)
    best_confidence = Column(Float, nullable=True)

    # Phase 6B: attributes attached at track level (summary of all frame attrs)
    attributes = Column(JSONB, nullable=True)
    # e.g. {"color": "red", "type": "sedan", "make": "Toyota"}

    # RAG text for this lifecycle event — embedded for search
    rag_text = Column(Text, nullable=True)

    created_at = Column(DateTime, default=datetime.utcnow)

    embedding = relationship(
        "TrackEventEmbedding",
        back_populates="track_event",
        uselist=False,
        cascade="all, delete-orphan",
    )


class TrackEventEmbedding(Base):
    """Vector embedding of TrackEvent.rag_text."""
    __tablename__ = "track_event_embeddings"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    track_event_id = Column(
        UUID(as_uuid=True),
        ForeignKey("track_events.id", ondelete="CASCADE"),
        nullable=False, unique=True, index=True,
    )
    embedding = Column(Vector(768), nullable=False)
    model_name = Column(String, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    track_event = relationship("TrackEvent", back_populates="embedding")


# ── Indexes ────────────────────────────────────────────────────────────────────
Index("idx_camera_time", Event.camera_id, Event.event_timestamp)
Index("idx_caption_video", Caption.video_filename, Caption.frame_second_offset)
Index("idx_summary_video", VideoSummary.video_filename, VideoSummary.camera_id)
Index("idx_processing_status_video", ProcessingStatus.video_filename, ProcessingStatus.status)
Index("idx_detected_objects_video_time", DetectedObject.video_filename, DetectedObject.frame_second_offset)
Index("idx_detected_objects_class", DetectedObject.video_filename, DetectedObject.object_class)
Index("idx_detected_objects_track", DetectedObject.video_filename, DetectedObject.track_id)
Index("idx_track_events_video", TrackEvent.video_filename, TrackEvent.track_id)
Index("idx_track_events_type", TrackEvent.video_filename, TrackEvent.event_type)