import re
import requests
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.core.logging import get_logger
from app.storage.models import Caption, VideoSummary, TrackEvent
from app.prompts.summary_prompt import SUMMARY_SYSTEM_PROMPT, SUMMARY_USER_TEMPLATE
from app.prompts.track_summary_prompt import TRACK_SUMMARY_SYSTEM_PROMPT, TRACK_SUMMARY_USER_TEMPLATE


def _extract_section(text: str, section: str) -> str:
    pattern = rf"{section}:\s*(.*?)(?=\n[A-Z ]+:|$)"
    match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return ""


def _condense_caption(caption_text: str) -> str:
    """
    Distill a full structured caption to core fields for temporal reasoning.
    Used for the legacy caption-based summarization path.
    """
    subjects = _extract_section(caption_text, "SUBJECTS")
    spatial = _extract_section(caption_text, "SPATIAL LAYOUT")
    anomalies = _extract_section(caption_text, "ANOMALIES")

    if not subjects and not spatial:
        return caption_text[:400]

    parts = []
    if subjects and subjects.lower() not in ("none observed.", "none observed"):
        parts.append(f"PRESENT: {subjects}")
    else:
        parts.append("PRESENT: Nothing/nobody")
    if spatial and spatial.lower() not in ("none observed.", "none observed"):
        parts.append(f"POSITION: {spatial}")
    if anomalies and anomalies.lower() not in ("none observed.", "none observed"):
        parts.append(f"ANOMALY: {anomalies}")

    return " | ".join(parts)


class VideoSummarizer:
    def __init__(self, db: Session):
        self.db = db
        self.settings = get_settings()
        self.logger = get_logger()
        self.model = self.settings.text_model

    # ── Phase 6A: Summary from structured track events ─────────────────────────

    def summarize_from_tracks(self, video_filename: str, force: bool = False) -> VideoSummary:
        """
        Generate a video summary from Phase 6A structured TrackEvent data.
        This replaces the caption-based summarizer for videos processed by
        VideoIntelligenceProcessor.

        The summary is grounded in deterministic detection data — no LLM
        hallucination about what objects were present. The LLM only interprets
        the confirmed detections.
        """
        # Fetch all track events for this video
        track_events = (
            self.db.query(TrackEvent)
            .filter(TrackEvent.video_filename == video_filename)
            .order_by(TrackEvent.first_seen_second)
            .all()
        )

        if not track_events:
            raise ValueError(
                f"No track events found for {video_filename}. "
                "Run VideoIntelligenceProcessor first."
            )

        camera_id = track_events[0].camera_id
        duration = max(e.last_seen_second for e in track_events)
        event_count = len(track_events)

        # Check cache — skip if summary already up to date
        existing = (
            self.db.query(VideoSummary)
            .filter(
                VideoSummary.video_filename == video_filename,
                VideoSummary.camera_id == camera_id,
            )
            .first()
        )
        if existing and existing.caption_count == event_count and not force:
            self.logger.info("track_summary_cache_hit", video=video_filename)
            return existing

        # Build structured event block for the LLM
        # Format: [Xs-Ys] CLASS #N: event_type (duration Ds)
        entry_events = [e for e in track_events if e.event_type == "entry"]
        exit_events = [e for e in track_events if e.event_type == "exit"]
        dwell_events = [e for e in track_events if e.event_type == "dwell"]

        def _fmt_events(events: list) -> str:
            if not events:
                return "  None"
            lines = []
            for e in events:
                # Include Phase 6B attributes if available
                attr_str = ""
                if e.attributes:
                    attrs = e.attributes
                    cls = attrs.get("object_class", e.object_class)
                    if cls in ("car", "truck", "bus", "motorcycle", "bicycle"):
                        color = attrs.get("color", "")
                        vtype = attrs.get("type", "")
                        make = attrs.get("make_estimate", "")
                        parts = [p for p in [color, vtype] if p and p != "unknown"]
                        if make and make != "unknown":
                            parts.append(f"({make})")
                        if parts:
                            attr_str = f" [{' '.join(parts)}]"
                    elif cls == "person":
                        gender = attrs.get("gender_estimate", "")
                        top = attrs.get("clothing_top", "")
                        bottom = attrs.get("clothing_bottom", "")
                        parts = [p for p in [gender, top, bottom] if p and p not in ("unknown", "none")]
                        if parts:
                            attr_str = f" [{', '.join(parts)}]"

                lines.append(
                    f"  [{e.first_seen_second:.1f}s-{e.last_seen_second:.1f}s] "
                    f"{e.object_class.upper()} track #{e.track_id}{attr_str} "
                    f"({e.duration_seconds:.1f}s)"
                )
            return "\n".join(lines)

        # Count unique objects by class
        unique_tracks = {}
        for e in entry_events:
            cls = e.object_class
            unique_tracks[cls] = unique_tracks.get(cls, 0) + 1

        objects_summary = ", ".join(
            f"{count} {cls}(s)" for cls, count in unique_tracks.items()
        ) or "no objects detected"

        event_block = (
            f"OBJECTS DETECTED: {objects_summary}\n\n"
            f"ENTRY EVENTS:\n{_fmt_events(entry_events)}\n\n"
            f"EXIT EVENTS:\n{_fmt_events(exit_events)}\n\n"
            f"DWELL EVENTS (prolonged presence):\n{_fmt_events(dwell_events)}"
        )

        user_message = TRACK_SUMMARY_USER_TEMPLATE.format(
            video_filename=video_filename,
            camera_id=camera_id,
            duration=duration,
            event_count=event_count,
            events=event_block,
        )

        payload = {
            "model": self.model,
            "system": TRACK_SUMMARY_SYSTEM_PROMPT,
            "prompt": user_message,
            "stream": False,
        }

        response = requests.post(
            f"{self.settings.ollama_host}/api/generate",
            json=payload,
            timeout=600,
        )
        response.raise_for_status()
        summary_text = response.json()["response"]

        # Save or update
        if existing:
            existing.summary_text = summary_text
            existing.caption_count = event_count
            existing.duration_seconds = duration
            existing.model_name = self.model
            existing.updated_at = __import__("datetime").datetime.utcnow()
            self.db.commit()
            self.db.refresh(existing)
            return existing
        else:
            summary = VideoSummary(
                video_filename=video_filename,
                camera_id=camera_id,
                summary_text=summary_text,
                caption_count=event_count,
                duration_seconds=duration,
                model_name=self.model,
            )
            self.db.add(summary)
            self.db.commit()
            self.db.refresh(summary)
            return summary

    # ── Legacy: Summary from captions (Phase 5 path, kept intact) ─────────────

    def summarize(self, video_filename: str, force: bool = False) -> VideoSummary:
        """
        Original caption-based summarizer. Used when Phase 5 caption data exists.
        Phase 6A calls summarize_from_tracks() instead.
        """
        captions = (
            self.db.query(Caption)
            .filter(Caption.video_filename == video_filename)
            .order_by(Caption.frame_second_offset)
            .all()
        )

        if not captions:
            raise ValueError(f"No captions found for video: {video_filename}")

        caption_count = len(captions)
        camera_id = captions[0].camera_id
        duration = captions[-1].frame_second_offset

        existing = (
            self.db.query(VideoSummary)
            .filter(
                VideoSummary.video_filename == video_filename,
                VideoSummary.camera_id == camera_id,
            )
            .first()
        )

        if existing and existing.caption_count == caption_count and not force:
            self.logger.info("summary_cache_hit", video=video_filename)
            return existing

        caption_lines = [
            f"[{c.frame_second_offset:.1f}s] {_condense_caption(c.caption_text)}"
            for c in captions
        ]
        captions_block = "\n".join(caption_lines)

        user_message = SUMMARY_USER_TEMPLATE.format(
            video_filename=video_filename,
            camera_id=camera_id,
            duration=duration,
            caption_count=caption_count,
            captions=captions_block,
        )

        payload = {
            "model": self.model,
            "system": SUMMARY_SYSTEM_PROMPT,
            "prompt": user_message,
            "stream": False,
        }

        response = requests.post(
            f"{self.settings.ollama_host}/api/generate",
            json=payload,
            timeout=600,
        )
        response.raise_for_status()
        summary_text = response.json()["response"]

        if existing:
            existing.summary_text = summary_text
            existing.caption_count = caption_count
            existing.duration_seconds = duration
            existing.model_name = self.model
            existing.updated_at = __import__("datetime").datetime.utcnow()
            self.db.commit()
            self.db.refresh(existing)
            return existing
        else:
            summary = VideoSummary(
                video_filename=video_filename,
                camera_id=camera_id,
                summary_text=summary_text,
                caption_count=caption_count,
                duration_seconds=duration,
                model_name=self.model,
            )
            self.db.add(summary)
            self.db.commit()
            self.db.refresh(summary)
            return summary

    def summarize_all(self, force: bool = False) -> list[VideoSummary]:
        videos = (
            self.db.query(Caption.video_filename)
            .distinct()
            .order_by(Caption.video_filename)
            .all()
        )
        results = []
        for (video_filename,) in videos:
            try:
                results.append(self.summarize(video_filename, force=force))
            except Exception as e:
                self.logger.error("summary_failed", video=video_filename, error=str(e))
        return results