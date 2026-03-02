import os
import signal
import numpy as np
import cv2

from app.core.config import get_settings
from app.core.logging import get_logger
from app.vision.frame_sampler import FrameSampler
from app.detection.detector import ObjectDetector
from app.detection.crop_utils import (
    extract_crop, save_crop, normalize_bbox, get_frame_quadrant,
)
from app.detection.event_generator import EventGenerator, TrackState, build_detection_rag_text
from app.detection.object_indexer import ObjectIndexer
from app.detection.attribute_processor import AttributeProcessor
from app.storage.database import SessionLocal
from app.storage.repository import EventRepository
from app.rag.summarizer import VideoSummarizer
from app.core.logging import get_logger


class VideoIntelligenceProcessor:
    """
    Phase 6A pipeline: YOLO detection + ByteTrack tracking on every frame.

    Flow per video:
      1. FrameSampler yields frames at configured FPS (e.g. 1 FPS)
      2. ObjectDetector.detect_with_tracking() runs YOLO + ByteTrack (~80ms/frame)
      3. Each detection is saved as a DetectedObject row with normalized bbox
      4. Crops of confident detections are saved to disk (for Phase 6B attributes)
      5. TrackState accumulated per track_id across the whole video
      6. After frame loop: EventGenerator produces entry/exit/dwell TrackEvents
      7. All DetectedObjects and TrackEvents are embedded into pgvector
      8. VideoSummarizer generates a text summary from structured track data

    Key difference from Phase 5 (SemanticVideoProcessor):
      - YOLO runs on EVERY sampled frame, not just scene changes
      - Scene change detector is removed as primary gate (YOLO is the gate now)
      - minicpm-v is NOT called in Phase 6A (saved for crop attribute extraction in 6B)
      - Processing time: ~80ms/frame instead of 3-5 min/frame
    """

    def __init__(self):
        self.settings = get_settings()
        self.logger = get_logger()
        self.detector = ObjectDetector()
        self.event_generator = EventGenerator(
            dwell_threshold_seconds=self.settings.dwell_threshold_seconds,
            exit_gap_seconds=self.settings.exit_gap_seconds,
        )
        self.base_data_dir = self.settings.video_input_path.rstrip("/")

        self._shutdown_requested = False
        self._current_video: str = None
        self._current_db = None

        signal.signal(signal.SIGTERM, self._handle_shutdown)
        signal.signal(signal.SIGINT, self._handle_shutdown)

    # ── Graceful shutdown ──────────────────────────────────────────────────────

    def _handle_shutdown(self, signum, frame_arg):
        self.logger.warning("shutdown_signal_received", signal=signum)
        self._shutdown_requested = True
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

    # ── Main pipeline ──────────────────────────────────────────────────────────

    def run(self):
        self.logger.info("intelligence_pipeline_started", data_dir=self.base_data_dir)

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

        # Load YOLO model + warm up once before any video
        self.detector.warmup()

        for video_file in video_files:
            if self._shutdown_requested:
                self.logger.warning("pipeline_stopping_shutdown_requested")
                break
            self._process_video(video_file)

        self.logger.info("intelligence_pipeline_completed")

    def _process_video(self, video_file: str):
        full_path = os.path.join(self.base_data_dir, video_file)

        db = SessionLocal()
        self._current_db = db
        self._current_video = video_file
        repo = EventRepository(db)
        indexer = ObjectIndexer(db)

        try:
            # ── Duplicate prevention ──────────────────────────────────────────
            if repo.is_completed(video_file):
                # Phase 6A done — but run 6B (attributes) if not yet extracted.
                # has_6b_completed is False on all existing videos (column defaults False).
                # This lets us extract attributes without re-running YOLO.
                if not repo.has_6b_completed(video_file) and repo.has_detection_data(video_file):
                    self.logger.info(
                        "video_6a_complete_running_6b_attributes", video=video_file,
                    )
                    self._run_6b_only(video_file, db, repo)
                else:
                    self.logger.info("video_already_completed_skipping", video=video_file)
                    repo.mark_skipped(video_file, self.settings.camera_id)
                return

            # ── Reset tracker state from any previous video ───────────────────
            self.detector.reset_tracker()

            # ── Mark running ──────────────────────────────────────────────────
            sampler = FrameSampler(full_path, self.settings.frame_sample_fps)
            repo.mark_running(
                video_file,
                self.settings.camera_id,
                total_frames=sampler.total_frames,
            )
            self.logger.info(
                "video_processing_started",
                video=video_file,
                total_frames=sampler.total_frames,
            )

            # ── Per-video state ───────────────────────────────────────────────
            track_states: dict[int, TrackState] = {}
            frames_processed = 0
            last_second = 0.0

            # ── Frame loop ────────────────────────────────────────────────────
            for frame, seconds in sampler:
                if self._shutdown_requested:
                    self.logger.warning("frame_loop_interrupted", video=video_file, second=seconds)
                    repo.mark_failed(video_file, "Interrupted mid-video by shutdown")
                    return

                last_second = seconds
                frames_processed += 1

                # Update progress every 5 frames (avoid hammering DB)
                if frames_processed % 5 == 0:
                    repo.mark_scene_detected(video_file, seconds)

                # ── YOLO + ByteTrack ──────────────────────────────────────────
                frame_dets = self._safe_detect(frame, seconds, video_file)
                if frame_dets is None:
                    continue

                if not frame_dets.has_objects:
                    continue

                self.logger.info(
                    "detections_found",
                    video=video_file,
                    second=seconds,
                    count=len(frame_dets.detections),
                    persons=frame_dets.person_count,
                    vehicles=frame_dets.vehicle_count,
                )

                # ── Process each detection ────────────────────────────────────
                for det in frame_dets.detections:
                    # Normalize bbox to 0-1
                    norm_x1, norm_y1, norm_x2, norm_y2 = normalize_bbox(
                        det.x1, det.y1, det.x2, det.y2,
                        frame_dets.frame_width, frame_dets.frame_height,
                    )

                    # Quadrant for spatial filtering
                    quadrant = get_frame_quadrant(
                        det.x1, det.y1, det.x2, det.y2,
                        frame_dets.frame_width, frame_dets.frame_height,
                    )

                    # Save crop for high-confidence detections (Phase 6B will use these)
                    crop_path = None
                    if det.confidence >= self.settings.crop_min_confidence and det.track_id is not None:
                        crop = extract_crop(frame, det)
                        if crop.size > 0:
                            crop_path = save_crop(
                                crop,
                                self.base_data_dir,
                                video_file,
                                seconds,
                                det.track_id,
                                det.object_class,
                            )

                    # Build RAG text for this detection
                    rag_text = build_detection_rag_text(
                        det.object_class, det.track_id, seconds,
                        det.confidence, quadrant,
                    )

                    # Save to DB
                    saved_obj = repo.save_detected_object(
                        video_filename=video_file,
                        camera_id=self.settings.camera_id,
                        frame_second_offset=seconds,
                        object_class=det.object_class,
                        confidence=det.confidence,
                        bbox_x1=norm_x1, bbox_y1=norm_y1,
                        bbox_x2=norm_x2, bbox_y2=norm_y2,
                        track_id=det.track_id,
                        frame_quadrant=quadrant,
                        crop_path=crop_path,
                        rag_text=rag_text,
                    )

                    # Index for semantic search
                    indexer.index_detected_object(saved_obj)

                    # Update track state for event generation
                    if det.track_id is not None:
                        self._update_track_state(
                            track_states, det, seconds, crop_path,
                        )

                # Update scenes_captioned counter (repurposed as "frames with detections")
                repo.mark_scene_captioned(video_file)

            # ── Generate lifecycle events ─────────────────────────────────────
            if not self._shutdown_requested and track_states:
                self.logger.info(
                    "generating_track_events",
                    video=video_file,
                    unique_tracks=len(track_states),
                )
                generated_events = self.event_generator.generate(track_states, last_second)

                for ev in generated_events:
                    saved_event = repo.save_track_event(
                        video_filename=video_file,
                        camera_id=self.settings.camera_id,
                        track_id=ev.track_id,
                        object_class=ev.object_class,
                        event_type=ev.event_type,
                        first_seen_second=ev.first_seen_second,
                        last_seen_second=ev.last_seen_second,
                        duration_seconds=ev.duration_seconds,
                        best_frame_second=ev.best_frame_second,
                        best_crop_path=ev.best_crop_path,
                        best_confidence=ev.best_confidence,
                        rag_text=ev.rag_text,
                    )
                    indexer.index_track_event(saved_event)

                self.logger.info(
                    "track_events_saved",
                    video=video_file,
                    count=len(generated_events),
                )

            # ── Phase 6B: Attribute extraction ───────────────────────────────
            # Runs after all track events are saved.
            # Calls minicpm-v on the best crop per unique track_id.
            # Upgrades rag_text with color/type/clothing attributes and re-embeds.
            if not self._shutdown_requested and track_states:
                self.logger.info("attribute_extraction_starting", video=video_file)
                try:
                    attr_processor = AttributeProcessor(db)
                    attributed = attr_processor.run(video_file)
                    repo.mark_6b_completed(video_file, attributed)
                    self.logger.info(
                        "attribute_extraction_complete",
                        video=video_file,
                        tracks_attributed=attributed,
                    )
                except Exception as e:
                    # Attribute extraction failure must NOT block summary or completion
                    self.logger.warning(
                        "attribute_extraction_failed",
                        video=video_file,
                        error=str(e),
                    )

            # ── Generate summary from structured track data ───────────────────
            if not self._shutdown_requested:
                self.logger.info("generating_video_summary", video=video_file)
                try:
                    VideoSummarizer(db).summarize_from_tracks(video_file)
                    self.logger.info("video_summary_complete", video=video_file)
                except Exception as e:
                    self.logger.warning(
                        "summary_generation_failed",
                        video=video_file,
                        error=str(e),
                    )

                repo.mark_completed(video_file)
                self.logger.info(
                    "video_processing_completed",
                    video=video_file,
                    frames_processed=frames_processed,
                    unique_tracks=len(track_states),
                )

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

    def _run_6b_only(self, video_file: str, db, repo) -> None:
        """
        Run Phase 6B attribute extraction on a video that already completed Phase 6A.
        Called when status=completed but phase_6b_completed=False.
        Does NOT re-run YOLO or re-process frames.
        """
        try:
            attr_processor = AttributeProcessor(db)
            attributed = attr_processor.run(video_file)
            repo.mark_6b_completed(video_file, attributed)
            self.logger.info(
                "6b_attributes_extracted",
                video=video_file,
                tracks_attributed=attributed,
            )
            # Regenerate summary with attribute-enriched rag_text
            try:
                from app.rag.summarizer import VideoSummarizer
                VideoSummarizer(db).summarize_from_tracks(video_file, force=True)
                self.logger.info("summary_regenerated_with_attributes", video=video_file)
            except Exception as e:
                self.logger.warning("summary_regen_failed", video=video_file, error=str(e))
        except Exception as e:
            self.logger.error("6b_only_run_failed", video=video_file, error=str(e))

    def _update_track_state(
        self,
        track_states: dict,
        det,
        seconds: float,
        crop_path,
    ):
        """Update or create TrackState for a track_id."""
        tid = det.track_id
        if tid not in track_states:
            track_states[tid] = TrackState(
                track_id=tid,
                object_class=det.object_class,
                first_seen=seconds,
                last_seen=seconds,
                frame_count=1,
                best_second=seconds,
                best_confidence=det.confidence,
                best_crop_path=crop_path,
                all_seconds=[seconds],
            )
        else:
            state = track_states[tid]
            state.last_seen = seconds
            state.frame_count += 1
            state.all_seconds.append(seconds)
            # Keep the highest-confidence crop as the "best" frame
            if det.confidence > state.best_confidence:
                state.best_confidence = det.confidence
                state.best_second = seconds
                if crop_path:
                    state.best_crop_path = crop_path

    def _safe_detect(self, frame, seconds: float, video_file: str):
        """
        Wrap YOLO inference with error recovery.
        Returns None on failure — frame is skipped, pipeline continues.
        """
        try:
            return self.detector.detect_with_tracking(frame, seconds)
        except Exception as e:
            self.logger.error(
                "detection_failed",
                video=video_file,
                second=seconds,
                error=str(e),
                action="skipping_frame_continuing",
            )
            return None