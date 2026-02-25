import os
from datetime import datetime

from app.core.config import get_settings
from app.vision.frame_sampler import FrameSampler
from app.vision.scene_change import SceneChangeDetector
from app.captioning.ollama_client import OllamaMultimodalClient
from app.storage.database import SessionLocal
from app.storage.repository import EventRepository
from app.core.logging import get_logger


class SemanticVideoProcessor:
    def __init__(self):
        self.settings = get_settings()
        self.logger = get_logger()

        self.scene_detector = SceneChangeDetector(
            self.settings.scene_change_threshold
        )

        self.captioner = OllamaMultimodalClient()

    def run(self):
        self.logger.info("semantic_pipeline_started")

        video_directory = "/app/app/data"

        if not os.path.exists(video_directory):
            self.logger.warning(
                "video_directory_not_found",
                path=video_directory
            )
            return

        video_files = [
            f for f in os.listdir(video_directory)
            if f.lower().endswith(".mp4")
        ]

        if not video_files:
            self.logger.warning("no_videos_found")
            return

        for video_file in video_files:
            full_path = os.path.join(video_directory, video_file)

            self.logger.info(
                "processing_video",
                video=video_file
            )

            try:
                sampler = FrameSampler(
                    full_path,
                    self.settings.frame_sample_fps,
                )

                for frame, seconds in sampler:

                    if self.scene_detector.is_scene_changed(frame):

                        self.logger.info(
                            "scene_change_detected",
                            video=video_file,
                            second=seconds
                        )

                        caption = self.captioner.generate_caption(frame)

                        db = SessionLocal()

                        try:
                            repo = EventRepository(db)

                            repo.save_caption(
                                camera_id=self.settings.camera_id,
                                video_filename=video_file,
                                frame_second_offset=seconds,
                                absolute_timestamp=None,
                                caption_text=caption,
                            )

                            self.logger.info(
                                "caption_stored",
                                video=video_file,
                                second=seconds,
                            )

                        finally:
                            db.close()

            except Exception as e:
                self.logger.error(
                    "video_processing_failed",
                    video=video_file,
                    error=str(e)
                )

        self.logger.info("semantic_pipeline_completed")