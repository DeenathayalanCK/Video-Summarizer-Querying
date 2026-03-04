"""
video_intelligence_processor.py — Phase 6A+6B+7A pipeline.

Bug fixes in this version:

BUG 1 — Pipeline backpressure
  Phase 6B (minicpm-v attribute extraction, ~30s/crop) used to block the
  entire pipeline.  Now runs in a background daemon thread.  The main thread
  starts summary generation immediately, then join()s the worker before
  calling mark_completed() to ensure all attributes are persisted.

BUG 2 — Frame-sequence motion reasoning
  TrackState.motion_samples collects (second, cx, cy, w, h) per frame.
  EventGenerator._analyse_motion() derives:
    standing / walking / running (from inter-frame displacement)
    fall_proxy, sudden_stop, direction_change (from bbox geometry signals)
  Results stored in TrackEvent.attributes["motion_summary"] and fed to
  TemporalAnalyzer for semantic behaviour classification.

BUG 3 — Appearance-based ReID for identity persistence
  AppearanceReID.embed_tracks() runs ResNet18 (torchvision, CPU) on each
  person's best crop → 512-dim L2-normalised vector.
  EventGenerator.generate(reid_embeddings=...) uses cosine similarity
  (threshold 0.75) to gate merge decisions between same-class tracks,
  replacing the blind class-only time-gap heuristic.
"""

import os
import signal
import threading
import numpy as np
import cv2

from app.core.config import get_settings
from app.core.logging import get_logger
from app.vision.frame_sampler import FrameSampler
from app.detection.detector import ObjectDetector
from app.detection.crop_utils import (
    extract_crop, save_crop, normalize_bbox, get_frame_quadrant,
)
from app.detection.event_generator import (
    EventGenerator, TrackState, MotionSample, build_detection_rag_text,
)
from app.detection.appearance_reid import AppearanceReID
from app.detection.temporal_analyzer import TemporalAnalyzer
from app.detection.object_indexer import ObjectIndexer
from app.detection.attribute_processor import AttributeProcessor
from app.storage.database import SessionLocal
from app.storage.repository import EventRepository
from app.rag.summarizer import VideoSummarizer
from app.detection.timeline_builder import TimelineBuilder

# Max time to wait for the Phase 6B worker thread before giving up
_6B_TIMEOUT_S = 3600   # 1 hour — minicpm-v on CPU is slow


class VideoIntelligenceProcessor:

    def __init__(self):
        self.settings = get_settings()
        self.logger = get_logger()
        self.detector = ObjectDetector()
        self.event_generator = EventGenerator(
            dwell_threshold_seconds=self.settings.dwell_threshold_seconds,
            exit_gap_seconds=self.settings.exit_gap_seconds,
            min_visible_frames=self.settings.min_visible_frames,
            merge_gap_seconds=self.settings.merge_gap_seconds,
        )
        self.reid = AppearanceReID()   # lazy-loads ResNet18 on first use
        self.base_data_dir = self.settings.video_input_path.rstrip("/")
        self._shutdown_requested = False
        self._current_video = None
        self._current_db = None

        signal.signal(signal.SIGTERM, self._handle_shutdown)
        signal.signal(signal.SIGINT, self._handle_shutdown)

    # ── Shutdown ───────────────────────────────────────────────────────────────

    def _handle_shutdown(self, signum, frame_arg):
        self.logger.warning("shutdown_signal_received", signal=signum)
        self._shutdown_requested = True
        if self._current_video and self._current_db:
            try:
                EventRepository(self._current_db).mark_failed(
                    self._current_video,
                    "Interrupted by shutdown signal — will resume on next run",
                )
            except Exception as e:
                self.logger.error("shutdown_save_failed", error=str(e))

    # ── Entry point ────────────────────────────────────────────────────────────

    def run(self):
        self.logger.info("pipeline_started", data_dir=self.base_data_dir)
        if not os.path.exists(self.base_data_dir):
            self.logger.warning("video_dir_missing", path=self.base_data_dir)
            return

        videos = sorted(f for f in os.listdir(self.base_data_dir)
                         if f.lower().endswith(".mp4"))
        if not videos:
            self.logger.warning("no_videos_found", path=self.base_data_dir)
            return

        self.logger.info("videos_found", count=len(videos))
        self.detector.warmup()

        for vf in videos:
            if self._shutdown_requested:
                break
            self._process_video(vf)

        self.logger.info("pipeline_complete")

    # ── Per-video pipeline ─────────────────────────────────────────────────────

    def _process_video(self, video_file: str):
        full_path = os.path.join(self.base_data_dir, video_file)
        db = SessionLocal()
        self._current_db = db
        self._current_video = video_file
        repo = EventRepository(db)
        indexer = ObjectIndexer(db)

        try:
            # ── Skip / resume logic ───────────────────────────────────────────
            if repo.is_completed(video_file):
                if not repo.has_6b_completed(video_file) and repo.has_detection_data(video_file):
                    self.logger.info("resuming_6b_only", video=video_file)
                    self._run_6b_only(video_file, db, repo)
                else:
                    self.logger.info("already_complete_skip", video=video_file)
                return

            if repo.has_detection_data(video_file) or repo.has_track_event_data(video_file):
                self.logger.warning("clearing_partial_data", video=video_file)
                repo.reset_for_rerun(video_file, self.settings.camera_id)

            self.detector.reset_tracker()

            sampler = FrameSampler(full_path, self.settings.frame_sample_fps)
            repo.mark_running(video_file, self.settings.camera_id,
                              total_frames=sampler.total_frames)
            self.logger.info("video_started", video=video_file,
                             total_frames=sampler.total_frames)

            # ── Per-video state ───────────────────────────────────────────────
            track_states: dict = {}
            frames_processed = 0
            last_second = 0.0

            # ── Frame loop ────────────────────────────────────────────────────
            for frame, seconds in sampler:
                if self._shutdown_requested:
                    repo.mark_failed(video_file, "Interrupted mid-video")
                    return

                last_second = seconds
                frames_processed += 1

                if frames_processed % 5 == 0:
                    repo.mark_scene_detected(video_file, seconds)

                frame_dets = self._safe_detect(frame, seconds, video_file)
                if not frame_dets or not frame_dets.has_objects:
                    continue

                self.logger.info(
                    "detections_found", video=video_file, second=seconds,
                    count=len(frame_dets.detections),
                    persons=frame_dets.person_count, vehicles=frame_dets.vehicle_count,
                )

                for det in frame_dets.detections:
                    nx1, ny1, nx2, ny2 = normalize_bbox(
                        det.x1, det.y1, det.x2, det.y2,
                        frame_dets.frame_width, frame_dets.frame_height,
                    )
                    quadrant = get_frame_quadrant(
                        det.x1, det.y1, det.x2, det.y2,
                        frame_dets.frame_width, frame_dets.frame_height,
                    )

                    crop_path = None
                    if (det.confidence >= self.settings.crop_min_confidence
                            and det.track_id is not None):
                        crop = extract_crop(frame, det)
                        if crop.size > 0:
                            crop_path = save_crop(
                                crop, self.base_data_dir, video_file,
                                seconds, det.track_id, det.object_class,
                            )

                    rag_text = build_detection_rag_text(
                        det.object_class, det.track_id, seconds, det.confidence, quadrant)

                    saved_obj = repo.save_detected_object(
                        video_filename=video_file,
                        camera_id=self.settings.camera_id,
                        frame_second_offset=seconds,
                        object_class=det.object_class,
                        confidence=det.confidence,
                        bbox_x1=nx1, bbox_y1=ny1, bbox_x2=nx2, bbox_y2=ny2,
                        track_id=det.track_id,
                        frame_quadrant=quadrant,
                        crop_path=crop_path,
                        rag_text=rag_text,
                    )
                    indexer.index_detected_object(saved_obj)

                    if det.track_id is not None:
                        # Bug 2: pass normalised bbox so motion samples are collected
                        self._update_track_state(
                            track_states, det, seconds, crop_path,
                            nx1, ny1, nx2, ny2,
                        )

                repo.mark_scene_captioned(video_file)

            # ── Bug 3: ReID embeddings for person tracks ──────────────────────
            reid_embeddings = {}
            if not self._shutdown_requested and track_states:
                self.logger.info("reid_starting", video=video_file)
                try:
                    reid_embeddings = self.reid.embed_tracks(track_states)
                    self.logger.info("reid_done", video=video_file,
                                     embedded=len(reid_embeddings))
                except Exception as e:
                    self.logger.warning("reid_failed", error=str(e))

            # ── Event generation (with ReID + motion analysis) ────────────────
            if not self._shutdown_requested and track_states:
                self.logger.info("generating_events", video=video_file,
                                 tracks=len(track_states))
                generated = self.event_generator.generate(
                    track_states, last_second,
                    reid_embeddings=reid_embeddings or None,
                )

                for ev in generated:
                    saved_ev = repo.save_track_event(
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
                    # Persist motion_summary + reid embedding into attributes
                    if ev.motion_summary:
                        attrs = dict(saved_ev.attributes or {})
                        attrs["motion_summary"] = ev.motion_summary
                        if ev.track_id in reid_embeddings:
                            # Store as list (JSON-serialisable)
                            attrs["reid_embedding"] = reid_embeddings[ev.track_id].tolist()
                        saved_ev.attributes = attrs
                    indexer.index_track_event(saved_ev)

                db.commit()
                self.logger.info("events_saved", video=video_file, count=len(generated))

            # ── Temporal behaviour analysis ───────────────────────────────────
            if not self._shutdown_requested and track_states:
                self.logger.info("temporal_starting", video=video_file)
                try:
                    from app.storage.models import TrackEvent, DetectedObject

                    all_dets = (db.query(DetectedObject)
                                .filter(DetectedObject.video_filename == video_file).all())
                    det_by_track = {}
                    for d in all_dets:
                        if d.track_id is not None:
                            det_by_track.setdefault(d.track_id, []).append(d)

                    behaviours = TemporalAnalyzer().analyze(track_states, det_by_track)
                    beh_map = {b.track_id: b for b in behaviours}

                    for ev in db.query(TrackEvent).filter(
                            TrackEvent.video_filename == video_file).all():
                        beh = beh_map.get(ev.track_id)
                        if beh:
                            attrs = dict(ev.attributes or {})
                            attrs["temporal"] = beh.to_dict()
                            ev.attributes = attrs
                    db.commit()
                    self.logger.info("temporal_done", video=video_file,
                                     tracks=len(behaviours))
                except Exception as e:
                    self.logger.warning("temporal_failed", error=str(e))

            # ── Bug 1: Phase 6B in background worker thread ───────────────────
            # Attribute extraction (minicpm-v, ~30s/crop) runs in a daemon thread
            # so summary generation can start immediately in the main thread.
            # We join() the worker before mark_completed() to ensure all
            # attribute data is persisted before the video is declared done.
            if not self._shutdown_requested and track_states:
                self.logger.info("6b_worker_starting", video=video_file)
                worker_exc = [None]

                def _run_6b():
                    worker_db = SessionLocal()
                    try:
                        ap = AttributeProcessor(worker_db)
                        n = ap.run(video_file)
                        EventRepository(worker_db).mark_6b_completed(video_file, n)
                        self.logger.info("6b_done", video=video_file, attributed=n)
                    except Exception as e:
                        worker_exc[0] = e
                        self.logger.warning("6b_failed", video=video_file, error=str(e))
                    finally:
                        worker_db.close()

                worker = threading.Thread(target=_run_6b, daemon=True, name="6b_worker")
                worker.start()

                # Summary runs in main thread while 6B runs in background
                self.logger.info("summary_starting", video=video_file)
                try:
                    VideoSummarizer(db).summarize_from_tracks(video_file)
                    self.logger.info("summary_done", video=video_file)
                except Exception as e:
                    self.logger.warning("summary_failed", video=video_file, error=str(e))

                # Build timeline graph + scene understanding
                self.logger.info("timeline_build_starting", video=video_file)
                try:
                    TimelineBuilder(db).build(video_file, self.settings.camera_id)
                    self.logger.info("timeline_built", video=video_file)
                except Exception as e:
                    self.logger.warning("timeline_build_failed", video=video_file, error=str(e))

                # Wait for 6B before completing
                worker.join(timeout=_6B_TIMEOUT_S)
                if worker.is_alive():
                    self.logger.warning("6b_worker_timed_out", video=video_file)

            else:
                # No tracks — just generate summary
                if not self._shutdown_requested:
                    try:
                        VideoSummarizer(db).summarize_from_tracks(video_file)
                    except Exception as e:
                        self.logger.warning("summary_failed", error=str(e))

            repo.mark_completed(video_file)
            self.logger.info("video_complete", video=video_file,
                             frames=frames_processed, tracks=len(track_states))

        except Exception as e:
            self.logger.error("video_failed", video=video_file, error=str(e))
            try:
                repo.mark_failed(video_file, str(e))
            except Exception:
                pass
        finally:
            db.close()
            self._current_db = None
            self._current_video = None

    # ── 6B-only path (already processed Phase 6A) ─────────────────────────────

    def _run_6b_only(self, video_file: str, db, repo):
        try:
            n = AttributeProcessor(db).run(video_file)
            repo.mark_6b_completed(video_file, n)
            self.logger.info("6b_only_done", video=video_file, attributed=n)
            try:
                VideoSummarizer(db).summarize_from_tracks(video_file, force=True)
            except Exception as e:
                self.logger.warning("summary_regen_failed", video=video_file, error=str(e))
        except Exception as e:
            self.logger.error("6b_only_failed", video=video_file, error=str(e))

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _update_track_state(
        self,
        track_states: dict,
        det,
        seconds: float,
        crop_path,
        nx1: float, ny1: float, nx2: float, ny2: float,
    ):
        """Create or update TrackState.  Appends a MotionSample for Bug 2."""
        tid = det.track_id
        cx = (nx1 + nx2) / 2
        cy = (ny1 + ny2) / 2
        w  = nx2 - nx1
        h  = ny2 - ny1
        ms = MotionSample(second=seconds, cx=cx, cy=cy, w=w, h=h)

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
                motion_samples=[ms],
            )
        else:
            st = track_states[tid]
            st.last_seen = seconds
            st.frame_count += 1
            st.all_seconds.append(seconds)
            st.motion_samples.append(ms)
            if det.confidence > st.best_confidence:
                st.best_confidence = det.confidence
                st.best_second = seconds
                if crop_path:
                    st.best_crop_path = crop_path

    def _safe_detect(self, frame, seconds: float, video_file: str):
        try:
            return self.detector.detect_with_tracking(frame, seconds)
        except Exception as e:
            self.logger.error("detect_failed", video=video_file,
                              second=seconds, error=str(e))
            return None