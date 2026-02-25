import uuid
from datetime import datetime
from sqlalchemy import (
    Column,
    String,
    Integer,
    Float,
    DateTime,
    Index,
    Text,
    ForeignKey,
)
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import declarative_base, relationship
from sqlalchemy.sql import func
from pgvector.sqlalchemy import Vector

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
    """
    Stores the vector embedding for each caption.
    Kept separate from captions so the embedding model can be swapped freely.
    """
    __tablename__ = "caption_embeddings"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    caption_id = Column(
        UUID(as_uuid=True),
        ForeignKey("captions.id", ondelete="CASCADE"),
        nullable=False,
        unique=True,
        index=True,
    )
    # nomic-embed-text produces 768-dim vectors
    embedding = Column(Vector(768), nullable=False)
    model_name = Column(String, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    caption = relationship("Caption", back_populates="embedding")


# Indexes
Index("idx_camera_time", Event.camera_id, Event.event_timestamp)
Index("idx_caption_video", Caption.video_filename, Caption.frame_second_offset)