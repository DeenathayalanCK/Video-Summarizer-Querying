import re
import requests
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.core.logging import get_logger
from app.storage.models import Caption, VideoSummary
from app.prompts.summary_prompt import SUMMARY_SYSTEM_PROMPT, SUMMARY_USER_TEMPLATE


def _extract_section(text: str, section: str) -> str:
    """
    Pull a specific section out of a structured caption.
    Returns the content after the section header up to the next header.
    """
    pattern = rf"{section}:\s*(.*?)(?=\n[A-Z ]+:|$)"
    match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return ""


def _condense_caption(caption_text: str) -> str:
    """
    Distill a full structured caption down to the 3 fields that matter
    for cross-caption temporal reasoning:
      - SUBJECTS (who/what is present)
      - SPATIAL LAYOUT (where they are)
      - ANOMALIES (anything unusual)

    This strips IMAGE QUALITY NOTES and verbose SCENE text that dilute
    the LLM's attention when comparing across many captions.
    """
    subjects = _extract_section(caption_text, "SUBJECTS")
    spatial = _extract_section(caption_text, "SPATIAL LAYOUT")
    anomalies = _extract_section(caption_text, "ANOMALIES")

    # If extraction failed (old-format captions), return truncated original
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

        self.logger.info("generating_summary", video=video_filename,
                         caption_count=caption_count, model=self.model)

        # Build condensed caption block for temporal reasoning
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
            result = existing
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
            result = summary

        self.logger.info("summary_generated", video=video_filename)
        return result

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