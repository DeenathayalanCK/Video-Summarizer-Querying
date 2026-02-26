import os
import signal
import numpy as np
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
        self.scene_detector = SceneChangeDetector(
            threshold=self.settings.scene_change_threshold,
            long_window_seconds=self.settings.scene_long_window_seconds,
            long_window_threshold=self.settings.scene_long_window_threshold,
            cooldown_seconds=self.settings.scene_cooldown_seconds,
        )
        self.captioner = OllamaMultimodalClient()
        self.base_data_dir = self.settings.video_input_path.rstrip("/")

        # Graceful shutdown — set by SIGTERM/SIGINT handler
        self._shutdown_requested = False
        self._current_video: str = None
        self._current_db = None

        signal.signal(signal.SIGTERM, self._handle_shutdown)
        signal.signal(signal.SIGINT, self._handle_shutdown)

    # ── Graceful shutdown ──────────────────────────────────────────────────────

    def _handle_shutdown(self, signum, frame):
        self.logger.warning("shutdown_signal_received", signal=signum)
        self._shutdown_requested = True

        # Mark currently-running video as failed so it re-processes on next run
        if self._current_video and self._current_db:
            try:
                repo = EventRepository(self._current_db)
                repo.mark_failed(
                    self._current_video,
                    error="Interrupted by shutdown signal — will resume on next run",
                )
                self.logger.info("shutdown_state_saved", video=self._current_video)
            except Exception as e:
                self.logger.error("shutdown_state_save_failed", error=str(e))

    # ── Model warm-up ──────────────────────────────────────────────────────────

    def _warmup_model(self):
        """
        Send a tiny blank frame to the vision model before processing starts.
        Forces Ollama to load model weights into RAM so the first real
        frame doesn't pay the cold-start penalty (4-8 min on CPU for minicpm-v).
        """
        self.logger.info("model_warmup_starting", model=self.settings.multimodal_model)
        try:
            # FIX: clean numpy warmup — no dead code, no inline __import__ hacks
            blank_frame = np.zeros((8, 8, 3), dtype=np.uint8)
            self.captioner.generate_caption(blank_frame)
            self.logger.info("model_warmup_complete", model=self.settings.multimodal_model)
        except Exception as e:
            # Warm-up failure is non-fatal — real processing continues
            self.logger.warning("model_warmup_failed", error=str(e))

    # ── Duplicate prevention ───────────────────────────────────────────────────

    def _should_skip(self, video_file: str, repo: EventRepository) -> bool:
        """
        Returns True if the video was fully processed in a prior run.
        Logs reason so operators can see what was skipped and why.
        """
        if repo.is_completed(video_file):
            self.logger.info(
                "video_already_completed_skipping",
                video=video_file,
                reason="status=completed in DB",
            )
            return True
        return False

    # ── Main pipeline ──────────────────────────────────────────────────────────

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

        # Warm up vision model once before any video processing starts
        self._warmup_model()

        for video_file in video_files:
            if self._shutdown_requested:
                self.logger.warning("pipeline_stopping_shutdown_requested")
                break

            self._process_video(video_file)

        self.logger.info("semantic_pipeline_completed")

    def _process_video(self, video_file: str):
        full_path = os.path.join(self.base_data_dir, video_file)
        self.scene_detector.reset()

        db = SessionLocal()
        self._current_db = db
        self._current_video = video_file
        repo = EventRepository(db)
        indexer = CaptionIndexer(db)

        try:
            # ── Duplicate prevention ──────────────────────────────────────────
            if self._should_skip(video_file, repo):
                repo.mark_skipped(video_file, self.settings.camera_id)
                return

            # ── Mark running ──────────────────────────────────────────────────
            sampler = FrameSampler(full_path, self.settings.frame_sample_fps)
            total_frames = sampler.total_frames
            repo.mark_running(
                video_file,
                self.settings.camera_id,
                total_frames=total_frames,
            )
            self.logger.info(
                "video_processing_started",
                video=video_file,
                total_frames=total_frames,
            )

            # ── Frame loop ────────────────────────────────────────────────────
            for frame, seconds in sampler:
                if self._shutdown_requested:
                    self.logger.warning("frame_loop_interrupted", video=video_file, second=seconds)
                    repo.mark_failed(video_file, "Interrupted mid-video by shutdown")
                    return

                if not self.scene_detector.is_scene_changed(frame, current_second=seconds):
                    continue

                # Scene change detected
                repo.mark_scene_detected(video_file, seconds)
                self.logger.info("scene_change_detected", video=video_file, second=seconds)

                keyframe_path = self.save_keyframe(frame, video_file, seconds)

                # ── Per-frame error recovery ──────────────────────────────────
                caption = self._safe_caption(frame, video_file, seconds)
                if caption is None:
                    # Already logged — continue to next scene instead of crashing
                    continue

                saved = repo.save_caption(
                    camera_id=self.settings.camera_id,
                    video_filename=video_file,
                    frame_second_offset=seconds,
                    absolute_timestamp=None,
                    keyframe_path=keyframe_path,
                    caption_text=caption,
                )
                repo.mark_scene_captioned(video_file)
                indexer.index_caption(saved)

                self.logger.info(
                    "caption_stored_and_indexed",
                    video=video_file,
                    second=seconds,
                )

            # ── Summarize ─────────────────────────────────────────────────────
            if not self._shutdown_requested:
                self.logger.info("generating_video_summary", video=video_file)
                VideoSummarizer(db).summarize(video_file)
                self.logger.info("video_summary_complete", video=video_file)

                repo.mark_completed(video_file)
                self.logger.info("video_processing_completed", video=video_file)

        except Exception as e:
            self.logger.error("video_processing_failed", video=video_file, error=str(e))
            try:
                repo.mark_failed(video_file, str(e))
            except Exception:
                pass
        finally:
            db.close()
            self._current_db = None
            self._current_video = None

    def _safe_caption(self, frame, video_file: str, seconds: float) -> str | None:
        """
        Attempt captioning with structured error recovery.
        Returns None on failure so the frame is skipped rather than crashing
        the whole video. Increments error_count in DB for observability.
        """
        try:
            return self.captioner.generate_caption(frame)
        except Exception as e:
            self.logger.error(
                "caption_generation_failed",
                video=video_file,
                second=seconds,
                error=str(e),
                action="skipping_frame_continuing",
            )
            db_temp = SessionLocal()
            try:
                EventRepository(db_temp).mark_failed(video_file, f"Frame {seconds}s: {e}")
            except Exception:
                pass
            finally:
                db_temp.close()
            return None

    def save_keyframe(self, frame, video_filename, second):
        keyframes_root = os.path.join(self.base_data_dir, "keyframes")
        video_stem = os.path.splitext(video_filename)[0]
        video_folder = os.path.join(keyframes_root, video_stem)
        os.makedirs(video_folder, exist_ok=True)
        filename = f"{round(second, 2)}.jpg"
        full_path = os.path.join(video_folder, filename)
        cv2.imwrite(full_path, frame)
        return full_path