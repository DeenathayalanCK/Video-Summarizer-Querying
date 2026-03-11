import re
import time
from datetime import datetime

import requests
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.core.logging import get_logger
from app.prompts.summary_prompt import SUMMARY_SYSTEM_PROMPT, SUMMARY_USER_TEMPLATE
from app.prompts.track_summary_prompt import TRACK_SUMMARY_SYSTEM_PROMPT, TRACK_SUMMARY_USER_TEMPLATE
from app.storage.models import Caption, TrackEvent, VideoSummary

_SUMMARY_TRACK_LIMIT = 8
_SUMMARY_PROMPT_CHAR_BUDGET = 2800
_TRACK_LINE_CHAR_LIMIT = 220


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


def _fmt_seconds(seconds: float) -> str:
    seconds = max(0, int(seconds))
    return f"{seconds // 60}:{seconds % 60:02d}"


def _truncate(text: str, limit: int) -> str:
    text = " ".join((text or "").split())
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 3)].rstrip(",; ") + "..."


class VideoSummarizer:
    def __init__(self, db: Session):
        self.db = db
        self.settings = get_settings()
        self.logger = get_logger()
        self.model = self.settings.summary_model
        self.num_ctx = self.settings.summary_num_ctx
        self.max_tokens = self.settings.summary_max_tokens

    def _call_generate(
        self,
        system_prompt: str,
        user_message: str,
        video_filename: str,
        *,
        prompt_kind: str,
        prompt_meta: dict | None = None,
    ) -> str:
        payload = {
            "model": self.model,
            "system": system_prompt,
            "prompt": user_message,
            "stream": False,
            "options": {
                "num_ctx": self.num_ctx,
                "num_predict": self.max_tokens,
            },
        }

        prompt_meta = dict(prompt_meta or {})
        prompt_meta.setdefault("prompt_chars", len(user_message))
        prompt_meta.setdefault("system_chars", len(system_prompt))
        prompt_meta.setdefault("num_ctx", self.num_ctx)
        prompt_meta.setdefault("max_tokens", self.max_tokens)
        prompt_meta.setdefault("model", self.model)
        start = time.monotonic()
        self.logger.info(
            "summarize_request_start",
            video=video_filename,
            prompt_kind=prompt_kind,
            **prompt_meta,
        )

        try:
            response = requests.post(
                f"{self.settings.ollama_host}/api/generate",
                json=payload,
                timeout=(10, self.settings.caption_timeout_seconds),
            )
            response.raise_for_status()
            summary_text = response.json().get("response", "").strip()
            if not summary_text:
                raise ValueError("empty response from Ollama")
            self.logger.info(
                "summarize_request_done",
                video=video_filename,
                prompt_kind=prompt_kind,
                elapsed_s=round(time.monotonic() - start, 1),
                response_chars=len(summary_text),
                **prompt_meta,
            )
            return summary_text
        except requests.exceptions.Timeout as exc:
            elapsed_s = round(time.monotonic() - start, 1)
            self.logger.warning(
                "summarize_timeout",
                video=video_filename,
                prompt_kind=prompt_kind,
                elapsed_s=elapsed_s,
                timeout_s=self.settings.caption_timeout_seconds,
                error_type=exc.__class__.__name__,
                **prompt_meta,
            )
            raise TimeoutError(
                f"Summary generation timed out after {elapsed_s}s for {video_filename}"
            ) from exc
        except requests.exceptions.RequestException as exc:
            elapsed_s = round(time.monotonic() - start, 1)
            self.logger.warning(
                "summarize_request_error",
                video=video_filename,
                prompt_kind=prompt_kind,
                elapsed_s=elapsed_s,
                error=str(exc),
                error_type=exc.__class__.__name__,
                **prompt_meta,
            )
            raise ValueError(
                f"Summary generation failed for {video_filename}: {exc}"
            ) from exc

    def _infer_duration(self, track_events: list[TrackEvent]) -> float:
        if not track_events:
            return 0.0
        first = min(e.first_seen_second for e in track_events)
        last = max(e.last_seen_second for e in track_events)

        if last > 1_000_000_000:
            return max(0.0, last - first)

        return max(0.0, last)

    def _build_track_line(
        self,
        entry_event: TrackEvent,
        exit_event: TrackEvent | None,
        dwell_event: TrackEvent | None,
    ) -> str:
        attrs = entry_event.attributes or {}
        parts = []
        if entry_event.object_class in ("car", "truck", "bus", "motorcycle", "bicycle"):
            for key in ("color", "type", "make_estimate"):
                value = attrs.get(key)
                if value and value not in ("unknown", "none", ""):
                    parts.append(str(value))
            plate = attrs.get("plate_number")
            if plate and plate not in ("unknown", ""):
                parts.append(f"plate {plate}")
        else:
            for key in (
                "person_label",
                "gender_estimate",
                "age_estimate",
                "clothing_top",
                "clothing_bottom",
                "head_covering",
                "carrying",
            ):
                value = attrs.get(key)
                if value and value not in ("unknown", "none", ""):
                    parts.append(str(value))
            activity = attrs.get("activity_caption") or attrs.get("activity_hint")
            if activity and activity not in ("unknown", "present", ""):
                parts.append(f"activity: {activity}")

        temporal = attrs.get("temporal") or {}
        behaviour = temporal.get("behaviour")
        if behaviour and behaviour not in ("unknown", "passing_through", ""):
            parts.append(f"behaviour: {behaviour}")

        motion_summary = attrs.get("motion_summary") or {}
        dominant = motion_summary.get("dominant_state")
        if dominant and dominant not in ("unknown", "stationary", ""):
            parts.append(f"motion: {dominant}")

        detail = ", ".join(parts) if parts else "no extra attributes"
        line = (
            f"- track #{entry_event.track_id}: {entry_event.object_class} | "
            f"seen {_fmt_seconds(entry_event.first_seen_second)}-{_fmt_seconds(entry_event.last_seen_second)} | "
            f"duration {entry_event.duration_seconds:.0f}s | {detail}"
        )
        if dwell_event:
            line += f" | dwell {_fmt_seconds(dwell_event.first_seen_second)}-{_fmt_seconds(dwell_event.last_seen_second)}"
        if exit_event:
            line += f" | exit {_fmt_seconds(exit_event.last_seen_second)}"
        return _truncate(line, _TRACK_LINE_CHAR_LIMIT)

    def _build_track_summary_block(
        self,
        entry_events: list[TrackEvent],
        exit_by_track: dict[int, TrackEvent],
        dwell_by_track: dict[int, TrackEvent],
    ) -> tuple[str, int, int]:
        ranked_entries = sorted(
            entry_events,
            key=lambda event: (
                event.duration_seconds,
                (event.best_confidence or 0.0),
            ),
            reverse=True,
        )

        lines = []
        used = 0
        omitted = 0
        used_chars = 0
        for event in ranked_entries:
            line = self._build_track_line(
                event,
                exit_by_track.get(event.track_id),
                dwell_by_track.get(event.track_id),
            )
            projected_chars = used_chars + len(line) + 1
            if used >= _SUMMARY_TRACK_LIMIT or projected_chars > _SUMMARY_PROMPT_CHAR_BUDGET:
                omitted += 1
                continue
            lines.append(line)
            used += 1
            used_chars = projected_chars

        if omitted:
            lines.append(f"- {omitted} additional shorter track(s) omitted for brevity")

        return "\n".join(lines), used, omitted

    def summarize_from_tracks(self, video_filename: str, force: bool = False) -> VideoSummary:
        track_events = (
            self.db.query(TrackEvent)
            .filter(TrackEvent.video_filename == video_filename)
            .order_by(TrackEvent.first_seen_second)
            .all()
        )
        if not track_events:
            raise ValueError(
                f"No track events found for {video_filename}. Run VideoIntelligenceProcessor first."
            )

        camera_id = track_events[0].camera_id
        duration = self._infer_duration(track_events)
        entry_events = [e for e in track_events if e.event_type == "entry"]
        event_count = len(entry_events)

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

        exit_by_track = {e.track_id: e for e in track_events if e.event_type == "exit"}
        dwell_by_track = {e.track_id: e for e in track_events if e.event_type == "dwell"}

        class_counts = {}
        for event in entry_events:
            class_counts[event.object_class] = class_counts.get(event.object_class, 0) + 1
        objects_summary = ", ".join(
            f"{count} {obj}(s)" for obj, count in sorted(class_counts.items())
        ) or "no objects detected"

        track_block, included_tracks, omitted_tracks = self._build_track_summary_block(
            entry_events,
            exit_by_track,
            dwell_by_track,
        )
        user_message = TRACK_SUMMARY_USER_TEMPLATE.format(
            video_filename=video_filename,
            camera_id=camera_id,
            duration=duration,
            duration_mm=_fmt_seconds(duration),
            event_count=event_count,
            events=(
                f"OBJECTS DETECTED: {objects_summary}\n"
                f"TRACKS INCLUDED: {included_tracks} of {len(entry_events)}\n\n"
                f"{track_block}"
            ),
        )

        summary_text = self._call_generate(
            TRACK_SUMMARY_SYSTEM_PROMPT,
            user_message,
            video_filename,
            prompt_kind="tracks",
            prompt_meta={
                "event_count": event_count,
                "included_tracks": included_tracks,
                "omitted_tracks": omitted_tracks,
                "duration_s": round(duration, 1),
            },
        )

        if existing:
            existing.summary_text = summary_text
            existing.caption_count = event_count
            existing.duration_seconds = duration
            existing.model_name = self.model
            existing.updated_at = datetime.utcnow()
            self.db.commit()
            self.db.refresh(existing)
            return existing

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

    def summarize(self, video_filename: str, force: bool = False) -> VideoSummary:
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
            for c in captions[:40]
        ]
        omitted_captions = max(0, len(captions) - len(caption_lines))
        if omitted_captions:
            caption_lines.append(
                f"[{duration:.1f}s] {omitted_captions} later caption(s) omitted for brevity"
            )
        captions_block = "\n".join(caption_lines)
        user_message = SUMMARY_USER_TEMPLATE.format(
            video_filename=video_filename,
            camera_id=camera_id,
            duration=duration,
            caption_count=caption_count,
            captions=captions_block,
        )
        summary_text = self._call_generate(
            SUMMARY_SYSTEM_PROMPT,
            user_message,
            video_filename,
            prompt_kind="captions",
            prompt_meta={
                "caption_count": caption_count,
                "included_captions": len(caption_lines),
                "omitted_captions": omitted_captions,
                "duration_s": round(duration, 1),
            },
        )

        if existing:
            existing.summary_text = summary_text
            existing.caption_count = caption_count
            existing.duration_seconds = duration
            existing.model_name = self.model
            existing.updated_at = datetime.utcnow()
            self.db.commit()
            self.db.refresh(existing)
            return existing

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
            except Exception as exc:
                self.logger.error("summary_failed", video=video_filename, error=str(exc))
        return results
