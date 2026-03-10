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
from sqlalchemy import text
from sqlalchemy.orm import Session
from sqlalchemy.orm.attributes import flag_modified

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
        # Track IDs already processed — skip duplicate entry events for the same track.
        # ByteTrack can re-assign the same integer ID within a window, producing
        # multiple entry rows.  Processing duplicates wastes LLM calls and can
        # cause re-embed conflicts; skip after the first successful attribution.
        already_attributed: set = set()

        for entry_event in entry_events:
            track_id = entry_event.track_id
            object_class = entry_event.object_class
            crop_path = entry_event.best_crop_path

            # Skip duplicate track_ids — already attributed in this run
            if track_id in already_attributed:
                continue

            try:
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

                # ── Skip if already attempted for this crop in a previous window ─────
                # Use "attr_attempted" sentinel written after any extraction attempt.
                # This prevents re-running minicpm-v every window even when it timed out.
                # A timed-out result writes sentinel but not meaningful attrs.
                existing_attrs = entry_event.attributes or {}
                if existing_attrs.get("attr_attempted"):
                    self.logger.info(
                        "attribute_processor_cache_hit",
                        video=video_filename,
                        track_id=track_id,
                        object_class=object_class,
                        has_data=existing_attrs.get("attr_has_data", False),
                    )
                    already_attributed.add(track_id)
                    attributed_count += 1
                    continue

                # ── Extract attributes ─────────────────────────────────────────────
                vehicle_attrs: Optional[VehicleAttributes] = None
                person_attrs: Optional[PersonAttributes] = None
                attributes_dict: dict = {}
                new_rag_text: str = entry_event.rag_text  # fallback = existing

                if object_class in VEHICLE_CLASSES:
                    vehicle_attrs = self.vehicle_extractor.extract(crop_path)
                    if not vehicle_attrs.has_data:
                        for ev in all_events_by_track.get(track_id, []):
                            merged = dict(ev.attributes or {})
                            merged["attr_attempted"] = True
                            merged["attr_has_data"] = False
                            merged["object_class"] = object_class
                            ev.attributes = merged
                            from sqlalchemy.orm.attributes import flag_modified as _fm
                            _fm(ev, "attributes")
                        self.db.commit()
                        already_attributed.add(track_id)
                        self.logger.info(
                            "attribute_processor_no_data",
                            video=video_filename,
                            track_id=track_id,
                            object_class=object_class,
                            note="model_timeout_or_all_unknown",
                        )
                        continue

                    attributes_dict = vehicle_attrs.to_dict()
                    attributes_dict["object_class"] = object_class
                    attributes_dict["attr_attempted"] = True
                    attributes_dict["attr_has_data"] = True

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
                        plate_number=vehicle_attrs.plate_number,
                    )

                elif object_class in PERSON_CLASSES:
                    person_attrs = self.person_extractor.extract(crop_path)
                    if not person_attrs.has_data:
                        # Extraction timed out or model returned all-unknown.
                        # Write sentinel so we don't retry on every window close,
                        # but skip writing all-unknown attrs (useless noise).
                        for ev in all_events_by_track.get(track_id, []):
                            merged = dict(ev.attributes or {})
                            merged["attr_attempted"] = True
                            merged["attr_has_data"] = False
                            merged["object_class"] = object_class
                            ev.attributes = merged
                            from sqlalchemy.orm.attributes import flag_modified as _fm
                            _fm(ev, "attributes")
                        self.db.commit()
                        already_attributed.add(track_id)
                        self.logger.info(
                            "attribute_processor_no_data",
                            video=video_filename,
                            track_id=track_id,
                            object_class=object_class,
                            note="model_timeout_or_all_unknown",
                        )
                        continue

                    attributes_dict = person_attrs.to_dict()
                    attributes_dict["object_class"] = object_class
                    attributes_dict["attr_attempted"] = True
                    attributes_dict["attr_has_data"] = True

                    new_rag_text = build_person_rag_text(
                        track_id=track_id,
                        event_type="entry",
                        first_seen=entry_event.first_seen_second,
                        last_seen=entry_event.last_seen_second,
                        duration=entry_event.duration_seconds,
                        confidence=entry_event.best_confidence or 0.0,
                        gender_estimate=person_attrs.gender_estimate,
                        age_estimate=person_attrs.age_estimate,
                        clothing_top=person_attrs.clothing_top,
                        clothing_bottom=person_attrs.clothing_bottom,
                        head_covering=person_attrs.head_covering,
                        carrying=person_attrs.carrying,
                        visible_text=person_attrs.visible_text,
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
                    # Merge new attributes into existing JSONB to preserve keys
                    # like "temporal" and "motion_summary" written by earlier pipeline
                    # phases.  Replacing entirely would wipe those values.
                    merged_attrs = dict(ev.attributes or {})
                    merged_attrs.update(attributes_dict)
                    ev.attributes = merged_attrs
                    flag_modified(ev, "attributes")

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
                            plate_number=vehicle_attrs.plate_number,
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
                            age_estimate=person_attrs.age_estimate,
                            clothing_top=person_attrs.clothing_top,
                            clothing_bottom=person_attrs.clothing_bottom,
                            head_covering=person_attrs.head_covering,
                            carrying=person_attrs.carrying,
                            visible_text=person_attrs.visible_text,
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

                # Keep-alive: the minicpm-v extraction above can take 30-90 s.
                # If the DB connection was idled out during that time, the flush
                # below would fail.  A lightweight SELECT 1 forces SQLAlchemy to
                # validate (and if needed, reconnect) before touching any data.
                try:
                    self.db.execute(text("SELECT 1"))
                except Exception:
                    self.db.rollback()

                self.db.flush()

                # ── Enrich rag_text with motion semantics before re-embedding ─────
                # Phase 6A wrote motion_clause into initial rag_text, but Phase 6B
                # overwrote rag_text with attribute-only text that drops motion events.
                # Re-append motion_summary so the embedding captures fall/stop/walking.
                for ev in track_events_for_this_track:
                    ms = (ev.attributes or {}).get("motion_summary", {})
                    mevts = ms.get("motion_events", [])
                    dom   = ms.get("dominant_state", "")
                    if ev.rag_text and (mevts or (dom and dom not in ("stationary", "unknown", ""))):
                        motion_parts = []
                        if dom and dom not in ("stationary", "unknown", ""):
                            motion_parts.append(f"predominantly {dom}")
                        if mevts:
                            motion_parts.append("; ".join(mevts[:3]))
                        ev.rag_text = ev.rag_text.rstrip(".") + ". Motion: " + ", ".join(motion_parts) + "."
                        flag_modified(ev, "rag_text")

                # NOTE: Embedding is handled by window_manager._run_reembed (step 6)
                # after ALL tracks are attributed. Embedding here would cause
                # duplicate Ollama calls (2×N instead of 1×N). Just commit attrs.
                self.db.commit()
                attributed_count += 1
                already_attributed.add(track_id)

                self.logger.info(
                    "attribute_processor_track_done",
                    video=video_filename,
                    track_id=track_id,
                    object_class=object_class,
                    attributed=True,
                )

            except Exception as track_exc:
                self.logger.warning(
                    "attribute_processor_track_failed",
                    video=video_filename,
                    track_id=entry_event.track_id,
                    object_class=entry_event.object_class,
                    error=str(track_exc),
                )
                # Roll back the failed transaction so the session remains usable
                # for subsequent tracks in this window.
                try:
                    self.db.rollback()
                except Exception:
                    pass

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