"""
live_models.py — Database models for live RTSP stream tracking.

Two tables:

  PersonIdentity — one row per KNOWN person (persistent across sessions).
    Stores their label (P1, P2...), appearance embedding, best face crop.
    Used for re-identification: new session matches against known identities.

  PersonSession — one row per visit/session.
    Stores entry time, exit time, last state, duration.
    Links to PersonIdentity via person_label.
    Wall-clock datetimes from the RTSP stream timestamp.
"""

import uuid
from datetime import datetime

from sqlalchemy import (
    Column, String, Float, Integer, DateTime,
    Boolean, Text, Index,
)
from sqlalchemy.dialects.postgresql import UUID, JSONB, ARRAY
from sqlalchemy.types import TypeDecorator, Float as SAFloat

from app.storage.database import Base


# ── PersonIdentity: known persons, persistent across sessions ─────────────────

class PersonIdentity(Base):
    """
    One row per unique known person.
    Created the first time a new person is seen.
    Updated when they are seen again (refresh embedding, update last_seen).
    """
    __tablename__ = "person_identities"

    id            = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    person_label  = Column(String(20), nullable=False, unique=True, index=True)
                   # e.g. "P1", "P2", "P3"

    # Appearance embedding (512-dim ResNet18 vector stored as JSONB float array)
    # Updated as a running average of all seen embeddings for this person
    embedding     = Column(JSONB, nullable=True)

    # Best face/body crop path for display
    best_crop_path = Column(String(500), nullable=True)

    # Cumulative stats
    total_visits  = Column(Integer, nullable=False, default=1)
    first_seen_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    last_seen_at  = Column(DateTime, nullable=False, default=datetime.utcnow)

    # Last known attributes from minicpm-v (clothing description etc.)
    attributes    = Column(JSONB, nullable=True)

    created_at    = Column(DateTime, default=datetime.utcnow)
    updated_at    = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


# ── PersonSession: one row per visit ─────────────────────────────────────────

class PersonSession(Base):
    """
    One row per person visit (entry → exit).
    entry_time is set when person is first detected.
    exit_time is updated every frame while person is visible.
    When person disappears for > LIVE_EXIT_TIMEOUT_SECONDS, session is closed.
    """
    __tablename__ = "person_sessions"

    id              = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    person_label    = Column(String(20), nullable=False, index=True)
                     # "P1", "P2" — links to PersonIdentity.person_label

    # ByteTrack track_id for this specific visit (resets between sessions)
    track_id        = Column(Integer, nullable=True)

    # Wall-clock times — from RTSP stream (datetime.utcnow() at detection time)
    entry_time      = Column(DateTime, nullable=False, default=datetime.utcnow)
    exit_time       = Column(DateTime, nullable=True)   # None = still present
    duration_seconds = Column(Float, nullable=True)     # computed on close

    # State tracking
    last_state      = Column(String(50), nullable=True)   # "walking", "stationary" etc.
    is_active       = Column(Boolean, nullable=False, default=True)
                    # True = currently in frame; False = exited

    # Best crop from this session
    best_crop_path  = Column(String(500), nullable=True)
    best_confidence = Column(Float, nullable=True)

    # Per-session attributes (may differ from identity — different clothing)
    attributes      = Column(JSONB, nullable=True)

    # Bounding box of last known position (pixel coords)
    last_bbox       = Column(JSONB, nullable=True)  # {x1,y1,x2,y2}

    created_at      = Column(DateTime, default=datetime.utcnow)
    updated_at      = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


Index("idx_person_sessions_label",   PersonSession.person_label)
Index("idx_person_sessions_active",  PersonSession.is_active)
Index("idx_person_sessions_entry",   PersonSession.entry_time)
Index("idx_person_identities_label", PersonIdentity.person_label)