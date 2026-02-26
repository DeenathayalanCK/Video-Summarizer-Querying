import os
import cv2

from app.core.config import get_settings
from app.vision.frame_sampler import FrameSampler
from app.vision.scene_change import SceneChangeDetector
from app.captioning.ollama_client import OllamaMultimodalClient
from app.storage.database import SessionLocal
from app.storage.repository import EventRepository
from app.rag.indexer import CaptionIndexer
from app.rag.summarizer import VideoSummarizer
from app.core.logging import get_logger


class SemanticVideoProcessor:
    def __init__(self):
        self.settings = get_settings()
        self.logger = get_logger()
        self.scene_detector = SceneChangeDetector(self.settings.scene_change_threshold)
        self.captioner = OllamaMultimodalClient()
        self.base_data_dir = self.settings.video_input_path.rstrip("/")

    def run(self):
        self.logger.info("semantic_pipeline_started", data_dir=self.base_data_dir)

        if not os.path.exists(self.base_data_dir):
            self.logger.warning("video_directory_not_found", path=self.base_data_dir)
            return

        video_files = sorted([
            f for f in os.listdir(self.base_data_dir)
            if f.lower().endswith(".mp4")
        ])

        if not video_files:
            self.logger.warning("no_videos_found", path=self.base_data_dir)
            return

        self.logger.info("videos_discovered", count=len(video_files), files=video_files)

        for video_file in video_files:
            full_path = os.path.join(self.base_data_dir, video_file)
            self.logger.info("processing_video", video=video_file)
            self.scene_detector.reset()

            try:
                sampler = FrameSampler(full_path, self.settings.frame_sample_fps)
                db = SessionLocal()
                repo = EventRepository(db)
                indexer = CaptionIndexer(db)

                try:
                    for frame, seconds in sampler:
                        if self.scene_detector.is_scene_changed(frame):
                            self.logger.info(
                                "scene_change_detected",
                                video=video_file,
                                second=seconds,
                            )

                            keyframe_path = self.save_keyframe(frame, video_file, seconds)
                            caption = self.captioner.generate_caption(frame)

                            saved = repo.save_caption(
                                camera_id=self.settings.camera_id,
                                video_filename=video_file,
                                frame_second_offset=seconds,
                                absolute_timestamp=None,
                                keyframe_path=keyframe_path,
                                caption_text=caption,
                            )

                            indexer.index_caption(saved)

                            self.logger.info(
                                "caption_stored_and_indexed",
                                video=video_file,
                                second=seconds,
                            )

                    # All frames done — auto-generate summary
                    self.logger.info("generating_video_summary", video=video_file)
                    summarizer = VideoSummarizer(db)
                    summarizer.summarize(video_file)
                    self.logger.info("video_summary_complete", video=video_file)

                finally:
                    db.close()

            except Exception as e:
                self.logger.error(
                    "video_processing_failed",
                    video=video_file,
                    error=str(e),
                )

        self.logger.info("semantic_pipeline_completed")

    def save_keyframe(self, frame, video_filename, second):
        keyframes_root = os.path.join(self.base_data_dir, "keyframes")
        video_stem = os.path.splitext(video_filename)[0]
        video_folder = os.path.join(keyframes_root, video_stem)
        os.makedirs(video_folder, exist_ok=True)
        filename = f"{round(second, 2)}.jpg"
        full_path = os.path.join(video_folder, filename)
        cv2.imwrite(full_path, frame)
        return full_path