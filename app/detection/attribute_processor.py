"""
Phase 6B: AttributeProcessor — batch attribute extraction after frame loop.

Runs AFTER the full frame loop and event generation complete.
Processes the best crop for each unique track_id, calls the appropriate
extractor (vehicle or person), writes attributes to DetectedObject +
TrackEvent rows, upgrades rag_text, and re-embeds.

Design:
- One minicpm-v call per unique track_id (not per frame)
- Writes attributes to TrackEvent.attributes (JSONB) for structured queries
- Writes to DetectedObject rows for the best frame (vehicle_color, etc.)
- Updates TrackEvent.rag_text with enriched text and re-embeds in pgvector
- Skips tracks with no crop saved (crop_min_confidence not met in 6A)
- Always re-processes (force=True behavior) — useful during development
"""

import os
from typing import Optional
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.core.logging import get_logger
from app.storage.models import DetectedObject, TrackEvent, TrackEventEmbedding
from app.detection.attribute_extractor import (
    VehicleAttributeExtractor,
    PersonAttributeExtractor,
    VehicleAttributes,
    PersonAttributes,
)
from app.prompts.attribute_prompt import (
    build_vehicle_rag_text,
    build_person_rag_text,
)
from app.rag.embedder import OllamaEmbedder

# Object classes handled by each extractor
VEHICLE_CLASSES = {"car", "truck", "bus", "motorcycle", "bicycle"}
PERSON_CLASSES = {"person"}


class AttributeProcessor:
    """
    Phase 6B batch processor.

    Called once per video after the Phase 6A frame loop completes.
    Finds the best crop per track, extracts attributes with minicpm-v,
    and upgrades the stored rag_text with enriched descriptions.

    Usage:
        processor = AttributeProcessor(db)
        processor.run(video_filename, track_states)
    """

    def __init__(self, db: Session):
        self.db = db
        self.settings = get_settings()
        self.logger = get_logger()
        self.embedder = OllamaEmbedder()
        self.vehicle_extractor = VehicleAttributeExtractor()
        self.person_extractor = PersonAttributeExtractor()

    def run(self, video_filename: str) -> int:
        """
        Process all TrackEvents for a video:
          1. For each unique track_id, find the event with a best_crop_path
          2. Extract attributes from the crop using minicpm-v
          3. Write attributes to TrackEvent.attributes (JSONB)
          4. Write class-specific fields to the best DetectedObject row
          5. Upgrade rag_text on all TrackEvents for this track_id
          6. Re-embed updated rag_text in pgvector

        Returns number of tracks successfully attributed.
        Always re-runs (no cache check) — always=True per project decision.
        """
        self.logger.info("attribute_processor_starting", video=video_filename)

        # Fetch all entry-type track events (one per track_id)
        # We use entry events as the canonical track record
        entry_events = (
            self.db.query(TrackEvent)
            .filter(
                TrackEvent.video_filename == video_filename,
                TrackEvent.event_type == "entry",
            )
            .order_by(TrackEvent.track_id)
            .all()
        )

        if not entry_events:
            self.logger.info(
                "attribute_processor_no_tracks",
                video=video_filename,
            )
            return 0

        # Also fetch exit and dwell events keyed by track_id for rag_text update
        all_events_by_track: dict[int, list[TrackEvent]] = {}
        all_track_events = (
            self.db.query(TrackEvent)
            .filter(TrackEvent.video_filename == video_filename)
            .all()
        )
        for ev in all_track_events:
            all_events_by_track.setdefault(ev.track_id, []).append(ev)

        attributed_count = 0

        for entry_event in entry_events:
            track_id = entry_event.track_id
            object_class = entry_event.object_class
            crop_path = entry_event.best_crop_path

            if not crop_path or not os.path.exists(crop_path):
                self.logger.info(
                    "attribute_processor_no_crop_skipping",
                    video=video_filename,
                    track_id=track_id,
                    object_class=object_class,
                    crop_path=crop_path,
                )
                continue

            self.logger.info(
                "attribute_processor_extracting",
                video=video_filename,
                track_id=track_id,
                object_class=object_class,
                crop=crop_path,
            )

            # ── Extract attributes ─────────────────────────────────────────────
            vehicle_attrs: Optional[VehicleAttributes] = None
            person_attrs: Optional[PersonAttributes] = None
            attributes_dict: dict = {}
            new_rag_text: str = entry_event.rag_text  # fallback = existing

            if object_class in VEHICLE_CLASSES:
                vehicle_attrs = self.vehicle_extractor.extract(crop_path)
                attributes_dict = vehicle_attrs.to_dict()
                attributes_dict["object_class"] = object_class

                new_rag_text = build_vehicle_rag_text(
                    track_id=track_id,
                    object_class=object_class,
                    event_type="entry",
                    first_seen=entry_event.first_seen_second,
                    last_seen=entry_event.last_seen_second,
                    duration=entry_event.duration_seconds,
                    confidence=entry_event.best_confidence or 0.0,
                    color=vehicle_attrs.color,
                    vehicle_type=vehicle_attrs.vehicle_type,
                    make_estimate=vehicle_attrs.make_estimate,
                )

            elif object_class in PERSON_CLASSES:
                person_attrs = self.person_extractor.extract(crop_path)
                attributes_dict = person_attrs.to_dict()
                attributes_dict["object_class"] = object_class

                new_rag_text = build_person_rag_text(
                    track_id=track_id,
                    event_type="entry",
                    first_seen=entry_event.first_seen_second,
                    last_seen=entry_event.last_seen_second,
                    duration=entry_event.duration_seconds,
                    confidence=entry_event.best_confidence or 0.0,
                    gender_estimate=person_attrs.gender_estimate,
                    clothing_top=person_attrs.clothing_top,
                    clothing_bottom=person_attrs.clothing_bottom,
                    head_covering=person_attrs.head_covering,
                    carrying=person_attrs.carrying,
                )
            else:
                self.logger.info(
                    "attribute_processor_unsupported_class",
                    object_class=object_class,
                    track_id=track_id,
                )
                continue

            # ── Write attributes to all TrackEvents for this track_id ──────────
            track_events_for_this_track = all_events_by_track.get(track_id, [])
            for ev in track_events_for_this_track:
                ev.attributes = attributes_dict

                # Build event-type-specific rag_text for exit/dwell
                if ev.event_type != "entry" and object_class in VEHICLE_CLASSES and vehicle_attrs:
                    ev.rag_text = build_vehicle_rag_text(
                        track_id=track_id,
                        object_class=object_class,
                        event_type=ev.event_type,
                        first_seen=ev.first_seen_second,
                        last_seen=ev.last_seen_second,
                        duration=ev.duration_seconds,
                        confidence=ev.best_confidence or 0.0,
                        color=vehicle_attrs.color,
                        vehicle_type=vehicle_attrs.vehicle_type,
                        make_estimate=vehicle_attrs.make_estimate,
                    )
                elif ev.event_type != "entry" and object_class in PERSON_CLASSES and person_attrs:
                    ev.rag_text = build_person_rag_text(
                        track_id=track_id,
                        event_type=ev.event_type,
                        first_seen=ev.first_seen_second,
                        last_seen=ev.last_seen_second,
                        duration=ev.duration_seconds,
                        confidence=ev.best_confidence or 0.0,
                        gender_estimate=person_attrs.gender_estimate,
                        clothing_top=person_attrs.clothing_top,
                        clothing_bottom=person_attrs.clothing_bottom,
                        head_covering=person_attrs.head_covering,
                        carrying=person_attrs.carrying,
                    )
                else:
                    # Entry event
                    ev.rag_text = new_rag_text

            # ── Write class-specific fields to the best DetectedObject ─────────
            if vehicle_attrs:
                self._update_best_detected_object_vehicle(
                    video_filename, track_id, vehicle_attrs,
                )
            elif person_attrs:
                self._update_best_detected_object_person(
                    video_filename, track_id, person_attrs,
                )

            self.db.flush()

            # ── Re-embed all TrackEvents for this track ────────────────────────
            for ev in track_events_for_this_track:
                self._reembed_track_event(ev)

            self.db.commit()
            attributed_count += 1

            self.logger.info(
                "attribute_processor_track_done",
                video=video_filename,
                track_id=track_id,
                object_class=object_class,
                attributed=True,
            )

        self.logger.info(
            "attribute_processor_complete",
            video=video_filename,
            attributed=attributed_count,
            total_tracks=len(entry_events),
        )
        return attributed_count

    def _update_best_detected_object_vehicle(
        self,
        video_filename: str,
        track_id: int,
        attrs: VehicleAttributes,
    ) -> None:
        """Write vehicle attributes to the highest-confidence DetectedObject for this track."""
        best = (
            self.db.query(DetectedObject)
            .filter(
                DetectedObject.video_filename == video_filename,
                DetectedObject.track_id == track_id,
            )
            .order_by(DetectedObject.confidence.desc())
            .first()
        )
        if best:
            best.vehicle_color = attrs.color if attrs.color != "unknown" else None
            best.vehicle_type = attrs.vehicle_type if attrs.vehicle_type != "unknown" else None
            best.vehicle_make = attrs.make_estimate if attrs.make_estimate != "unknown" else None

    def _update_best_detected_object_person(
        self,
        video_filename: str,
        track_id: int,
        attrs: PersonAttributes,
    ) -> None:
        """Write person attributes to the highest-confidence DetectedObject for this track."""
        best = (
            self.db.query(DetectedObject)
            .filter(
                DetectedObject.video_filename == video_filename,
                DetectedObject.track_id == track_id,
            )
            .order_by(DetectedObject.confidence.desc())
            .first()
        )
        if best:
            best.person_gender = attrs.gender_estimate if attrs.gender_estimate != "unknown" else None
            best.person_clothing_top = attrs.clothing_top if attrs.clothing_top != "unknown" else None
            best.person_clothing_bottom = attrs.clothing_bottom if attrs.clothing_bottom != "unknown" else None

    def _reembed_track_event(self, event: TrackEvent) -> None:
        """
        Re-embed a TrackEvent's rag_text.
        Deletes existing embedding and creates a fresh one.
        This is the correct approach when rag_text has been upgraded.
        """
        if not event.rag_text:
            return

        try:
            # Delete existing embedding
            existing = (
                self.db.query(TrackEventEmbedding)
                .filter(TrackEventEmbedding.track_event_id == event.id)
                .first()
            )
            if existing:
                self.db.delete(existing)
                self.db.flush()

            # Create fresh embedding with upgraded rag_text
            vector = self.embedder.embed(event.rag_text)
            new_emb = TrackEventEmbedding(
                track_event_id=event.id,
                embedding=vector,
                model_name=self.settings.embed_model,
            )
            self.db.add(new_emb)

        except Exception as e:
            self.logger.warning(
                "track_event_reembed_failed",
                event_id=str(event.id),
                error=str(e),
            )