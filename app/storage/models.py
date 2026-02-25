import uuid
from datetime import datetime
from sqlalchemy import (
    Column,
    String,
    Integer,
    Float,
    DateTime,
    Index,
    Text
)
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import declarative_base
from sqlalchemy.sql import func


Base = declarative_base()


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

    created_at = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )

class Caption(Base):
    __tablename__ = "captions"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    camera_id = Column(String, nullable=False)
    video_filename = Column(String, nullable=False)
    frame_second_offset = Column(Float, nullable=False)
    absolute_timestamp = Column(DateTime, nullable=True)
    caption_text = Column(Text, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    keyframe_path = Column(String, nullable=False)
    
# Composite index for faster camera + time queries
Index(
    "idx_camera_time",
    Event.camera_id,
    Event.event_timestamp,
)