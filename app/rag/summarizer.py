import requests
from sqlalchemy.orm import Session
from sqlalchemy import func

from app.core.config import get_settings
from app.core.logging import get_logger
from app.storage.models import Caption, VideoSummary
from app.prompts.summary_prompt import SUMMARY_SYSTEM_PROMPT, SUMMARY_USER_TEMPLATE


class VideoSummarizer:
    """
    Generates a structured summary for a video by passing all its
    chronologically-ordered captions to the text LLM.

    Summaries are cached in the VideoSummary table and only regenerated
    when the caption count changes (i.e. new frames were indexed).
    """

    def __init__(self, db: Session):
        self.db = db
        self.settings = get_settings()
        self.logger = get_logger()
        self.model = self.settings.text_model

    def summarize(self, video_filename: str, force: bool = False) -> VideoSummary:
        """
        Generate and store a summary for the given video.
        Returns cached summary if caption count hasn't changed, unless force=True.
        """
        # Fetch all captions in chronological order
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

        # Check for existing summary
        existing = (
            self.db.query(VideoSummary)
            .filter(
                VideoSummary.video_filename == video_filename,
                VideoSummary.camera_id == camera_id,
            )
            .first()
        )

        # Return cached if caption count unchanged and not forced
        if existing and existing.caption_count == caption_count and not force:
            self.logger.info(
                "summary_cache_hit",
                video=video_filename,
                caption_count=caption_count,
            )
            return existing

        self.logger.info(
            "generating_summary",
            video=video_filename,
            caption_count=caption_count,
            model=self.model,
        )

        # Build caption block
        caption_lines = [
            f"[{c.frame_second_offset:.1f}s]\n{c.caption_text}"
            for c in captions
        ]
        captions_block = "\n\n".join(caption_lines)

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
            timeout=300,   # summaries can be slow for long videos
        )
        response.raise_for_status()

        summary_text = response.json()["response"]

        # Upsert — update if exists, insert if not
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

        self.logger.info(
            "summary_generated",
            video=video_filename,
            caption_count=caption_count,
        )

        return result

    def summarize_all(self, force: bool = False) -> list[VideoSummary]:
        """Generate summaries for every video that has captions."""
        videos = (
            self.db.query(Caption.video_filename)
            .distinct()
            .order_by(Caption.video_filename)
            .all()
        )

        results = []
        for (video_filename,) in videos:
            try:
                summary = self.summarize(video_filename, force=force)
                results.append(summary)
            except Exception as e:
                self.logger.error(
                    "summary_failed",
                    video=video_filename,
                    error=str(e),
                )

        return results