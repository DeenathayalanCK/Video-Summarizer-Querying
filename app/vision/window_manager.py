"""
window_manager.py — 5-minute rolling window lifecycle for live RTSP streams.

Every LIVE_WINDOW_MINUTES (default 5), closes the current window and opens a
new one. On close, queues attribute extraction → temporal analysis → summary
in a background worker thread so the MJPEG stream never blocks.

Window key format:
    {camera_id}_{YYYYMMDD}_{HHMM}
    e.g.  gate_cam_01_20260305_1600

This key is used as `video_filename` in ALL existing DB tables:
  TrackEvent, DetectedObject, ProcessingStatus, VideoSummary, SemanticMemoryGraph
So every existing RAG query, Ask, Search, Timeline, Temporal, Summary tab works
against live windows identically to recorded videos — zero code changes needed.

Cross-boundary tracks:
  A person entering at 15:59 and exiting at 16:03 appears in BOTH windows.
  The LiveStreamProcessor calls window_manager.current_window_key() every frame.
  When the key changes, it closes/flushes the current track into the old window
  AND re-opens it in the new window. This is handled in live_stream_processor.py.
"""

import threading
import time
from datetime import datetime, timedelta
from typing import Optional

from app.core.config import get_settings
from app.core.logging import get_logger


class WindowManager:
    """
    Singleton. Manages the current 5-minute window key and triggers
    post-processing (attributes → temporal → summary) when a window closes.
    """
    _instance: Optional["WindowManager"] = None
    _cls_lock  = threading.Lock()

    @classmethod
    def get_instance(cls) -> "WindowManager":
        with cls._cls_lock:
            if cls._instance is None:
                cls._instance = cls()
            return cls._instance

    def __init__(self):
        self.settings = get_settings()
        self.logger   = get_logger()

        self._lock           = threading.RLock()
        self._current_key:   Optional[str] = None
        self._window_start:  Optional[datetime] = None
        self._thread:        Optional[threading.Thread] = None
        self._stop_evt       = threading.Event()
        self._process_queue  = []  # list of window keys awaiting post-processing

        # Semaphore: only ONE postprocess thread may call Ollama at a time.
        # Two concurrent windows fight for the single CPU Ollama slot, causing
        # both to timeout.  With semaphore, thread B waits for thread A to finish
        # BEFORE calling Ollama — so each gets 100% of Ollama CPU time.
        # Steps that don't call Ollama (temporal, timeline, memory, embeddings)
        # are NOT gated — only the attribute-extraction step acquires the sem.
        self._ollama_sem = threading.Semaphore(1)
        self._ask_pending = threading.Event()

    # ── Public API ────────────────────────────────────────────────────────────

    def start(self):
        """Start the window rotation thread."""
        self._open_window(datetime.utcnow())
        self._stop_evt.clear()
        self._thread = threading.Thread(
            target=self._rotation_loop, daemon=True, name="window_manager")
        self._thread.start()
        self.logger.info("window_manager_started",
                         window=self._current_key,
                         interval_minutes=self.settings.live_window_minutes)

    def stop(self):
        """Stop the rotation loop. Closes current window and queues its post-processing."""
        self._stop_evt.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=15)
        final_key = None
        with self._lock:
            if self._current_key:
                final_key = self._current_key
                self._close_window(self._current_key)
        # Trigger post-processing for the final (possibly partial) window
        if final_key:
            self._queue_postprocess(final_key)
        self.logger.info("window_manager_stopped")

    @property
    def current_key(self) -> Optional[str]:
        """The window key for the current time slot. Thread-safe."""
        with self._lock:
            return self._current_key

    @property
    def current_window_start(self) -> Optional[datetime]:
        with self._lock:
            return self._window_start

    def key_for_time(self, dt: datetime) -> str:
        """
        Compute the window key for any given datetime.
        Rounds down to the nearest LIVE_WINDOW_MINUTES boundary.
        """
        mins = self.settings.live_window_minutes
        floored_minute = (dt.minute // mins) * mins
        boundary = dt.replace(minute=floored_minute, second=0, microsecond=0)
        return f"{self.settings.camera_id}_{boundary.strftime('%Y%m%d_%H%M')}"

    def window_start_for_key(self, key: str) -> Optional[datetime]:
        """Parse a window key back to its start datetime."""
        try:
            # key = camera_id_YYYYMMDD_HHMM
            # camera_id may contain underscores, so split from right
            parts = key.rsplit("_", 2)  # ['gate_cam_01', '20260305', '1600']
            if len(parts) == 3:
                return datetime.strptime(parts[1] + parts[2], "%Y%m%d%H%M")
        except Exception:
            pass
        return None

    # ── Rotation loop ─────────────────────────────────────────────────────────

    def _rotation_loop(self):
        """
        Wakes up every 10 seconds, checks if the current window has expired,
        and rotates to a new one if so.
        """
        while not self._stop_evt.is_set():
            self._stop_evt.wait(timeout=10)
            if self._stop_evt.is_set():
                break
            now = datetime.utcnow()
            expected_key = self.key_for_time(now)
            with self._lock:
                if self._current_key != expected_key:
                    old_key = self._current_key
                    self._close_window(old_key)
                    self._open_window(now)
                    # Queue post-processing for the closed window
                    if old_key:
                        self._queue_postprocess(old_key)

    def _open_window(self, dt: datetime):
        key = self.key_for_time(dt)
        with self._lock:
            self._current_key  = key
            self._window_start = dt.replace(
                minute=(dt.minute // self.settings.live_window_minutes)
                        * self.settings.live_window_minutes,
                second=0, microsecond=0,
            )
        self.logger.info("window_opened", key=key)
        # Create ProcessingStatus row so sidebar shows this window immediately
        self._create_processing_status(key)

    def _close_window(self, key: str):
        if not key:
            return
        self.logger.info("window_closing", key=key)
        # Mark window as "post_processing" in DB
        self._mark_status(key, "post_processing")

    def _queue_postprocess(self, key: str):
        """
        Spawn a background thread to run attribute extraction → temporal → summary.
        Non-blocking: stream continues while this runs.
        The shared _ollama_sem ensures only one thread calls Ollama at a time.
        """
        t = threading.Thread(
            target=self._run_postprocess,
            args=(key,),
            daemon=True,
            name=f"postprocess_{key}",
        )
        t.start()

    def _pipeline_ollama_ctx(self, key: str, step_name: str):
        """
        Context manager for pipeline Ollama calls with Ask priority.
        If a user Ask is pending AND Ollama is busy, skip this pipeline step so
        the Ask gets Ollama access first. Step runs on the next window instead.
        Yields True if sem acquired (proceed), False if deferred (skip).
        """
        import contextlib

        @contextlib.contextmanager
        def _ctx():
            if self._ask_pending.is_set():
                got = self._ollama_sem.acquire(blocking=False)
                if not got:
                    self.logger.info(
                        "pipeline_step_deferred_for_ask",
                        key=key, step=step_name,
                    )
                    yield False
                    return
            else:
                self._ollama_sem.acquire(blocking=True)
                got = True
            try:
                yield got
            finally:
                if got:
                    self._ollama_sem.release()

        return _ctx()

    def _run_postprocess(self, key: str):
        """
        Sequential post-processing for a closed window.
        Pipeline: attributes → temporal → timeline → memory → activity → reembed → summary
        Each step updates ProcessingStatus.current_step so the UI can show live progress.
        """
        import time
        from app.storage.database import SessionLocal
        db = SessionLocal()
        _t0 = time.monotonic()
        def _elapsed():
            return round(time.monotonic() - _t0, 1)
        try:
            # Brief delay to let LiveStreamProcessor flush final frame writes
            time.sleep(3)
            self.logger.info("window_postprocess_start", key=key)

            # ── STEP 1: Attribute extraction ─────────────────────────────────
            # Calls Ollama vision model per track.
            # Semaphore ensures only ONE window's postprocess thread calls Ollama
            # at a time — prevents thread contention that caused all-timeout failures.
            self._set_step(key, "🧠 Attributes", db)
            try:
                from app.core.config import get_settings as _gs_attr
                if _gs_attr().multimodal_model:
                    self.logger.info("window_attrs_waiting_sem", key=key)
                    with self._pipeline_ollama_ctx(key, "attrs") as _sem_ok:
                        if not _sem_ok:
                            return
                        self.logger.info("window_attrs_sem_acquired", key=key)
                        from app.detection.attribute_processor import AttributeProcessor
                        n_attr = AttributeProcessor(db).run(key)
                    self.logger.info("window_attrs_done", key=key, tracks=n_attr, elapsed_s=_elapsed())
                else:
                    self.logger.info("window_attrs_skipped",
                                     reason="no_multimodal_model", key=key)
            except Exception as e:
                self.logger.warning("window_attrs_failed", key=key, error=str(e))

            # ── STEP 2: Temporal analysis (with MotionSample reconstruction) ─
            self._set_step(key, "📊 Temporal", db)
            try:
                self._run_temporal(key, db)
                self.logger.info("window_temporal_done", key=key, elapsed_s=_elapsed())
            except Exception as e:
                self.logger.warning("window_temporal_failed", key=key, error=str(e))

            # ── STEP 3: Timeline builder ─────────────────────────────────────
            self._set_step(key, "📅 Timeline", db)
            try:
                from app.detection.timeline_builder import TimelineBuilder
                from app.core.config import get_settings
                TimelineBuilder(db).build(key, get_settings().camera_id)
                self.logger.info("window_timeline_done", key=key, elapsed_s=_elapsed())
            except Exception as e:
                self.logger.warning("window_timeline_failed", key=key, error=str(e))

            # ── STEP 4: Memory graph ─────────────────────────────────────────
            self._set_step(key, "🕸 Memory", db)
            try:
                from app.storage.memory_graph import MemoryGraphBuilder
                from app.core.config import get_settings
                MemoryGraphBuilder(db).build(key, get_settings().camera_id)
                self.logger.info("window_memory_graph_done", key=key, elapsed_s=_elapsed())
            except Exception as e:
                self.logger.warning("window_memory_graph_failed", key=key, error=str(e))

            # ── STEP 5: Activity captions (minicpm-v per track) ──────────────
            # Also gated by semaphore — calls Ollama vision model same as Step 1.
            # Without gating, Step 5 of window N runs simultaneously with Step 7
            # (summary) of window N-1, causing both to timeout from RAM pressure.
            self._set_step(key, "🎯 Activity AI", db)
            try:
                from app.core.config import get_settings as _gs_act
                if _gs_act().multimodal_model:
                    self.logger.info("window_activity_waiting_sem", key=key)
                    with self._pipeline_ollama_ctx(key, "activity") as _sem_ok:
                        if not _sem_ok:
                            return
                        self.logger.info("window_activity_sem_acquired", key=key)
                        from app.detection.activity_detector import run_activity_captions_for_window
                        n_cap = run_activity_captions_for_window(key, db)
                    self.logger.info("window_activity_captions_done", key=key, captioned=n_cap)
                else:
                    self.logger.info("window_activity_captions_skipped",
                                     reason="no_multimodal_model", key=key)
            except Exception as e:
                self.logger.warning("window_activity_captions_failed", key=key, error=str(e))

            # ── STEP 6: Re-embed updated rag_text ───────────────────────────
            self._set_step(key, "🔢 Embeddings", db)
            try:
                self._run_reembed(key, db)
                self.logger.info("window_reembed_done", key=key, elapsed_s=_elapsed())
            except Exception as e:
                self.logger.warning("window_reembed_failed", key=key, error=str(e))

            # ── STEP 7: Summary ──────────────────────────────────────────────
            # Gated by semaphore: llama3.2 summary must NOT overlap with moondream
            # attribute extraction from a concurrent window's Step 1 or Step 5.
            # Without gating: llama3.2 (2.0GB) + moondream (1.8GB) + both KV caches
            # saturate RAM → OS pages both out → summary times out at 300s every time.
            # With gating: llama3.2 gets full CPU alone → completes in ~60-90s.
            self._set_step(key, "📝 Summary", db)
            try:
                self.logger.info("window_summary_waiting_sem", key=key)
                with self._pipeline_ollama_ctx(key, "summary") as _sem_ok:
                    if not _sem_ok:
                        raise Exception("deferred_for_ask")
                    self.logger.info("window_summary_sem_acquired", key=key)
                    from app.rag.summarizer import VideoSummarizer
                    VideoSummarizer(db).summarize_from_tracks(key)
                self.logger.info("window_summary_done", key=key, elapsed_s=_elapsed())
            except Exception as e:
                self.logger.warning("window_summary_failed", key=key, error=str(e))

            # ── Final: write counts to ProcessingStatus ──────────────────────
            try:
                from app.storage.models import TrackEvent, DetectedObject, ProcessingStatus
                n_tracks = db.query(TrackEvent).filter(
                    TrackEvent.video_filename == key,
                    TrackEvent.event_type == "entry",
                ).count()
                n_dets = db.query(DetectedObject).filter(
                    DetectedObject.video_filename == key,
                ).count()
                ps = db.query(ProcessingStatus).filter(
                    ProcessingStatus.video_filename == key).first()
                if ps:
                    ps.scenes_detected   = n_tracks
                    ps.scenes_captioned  = n_dets
                    ps.phase_6b_completed = True
                    ps.phase_6b_tracks_attributed = n_tracks
                    ps.current_step = None   # clear step label
                    db.commit()
            except Exception as e:
                self.logger.warning("window_status_counts_failed", key=key, error=str(e))

            self._mark_status(key, "completed")
            self.logger.info("window_postprocess_complete", key=key, total_s=_elapsed())

        except Exception as e:
            self.logger.error("window_postprocess_error", key=key, error=str(e))
            self._mark_status(key, "failed")
        finally:
            db.close()

    def _set_step(self, key: str, label: str, db):
        """Update ProcessingStatus.current_step for live pipeline progress display."""
        try:
            from app.storage.models import ProcessingStatus
            ps = db.query(ProcessingStatus).filter(
                ProcessingStatus.video_filename == key).first()
            if ps:
                ps.current_step = label
                db.commit()
            self.logger.info("pipeline_step", key=key, step=label)
        except Exception:
            pass

    def _run_temporal(self, key: str, db):
        """
        Reconstruct TrackState with MotionSamples from DB, then run TemporalAnalyzer.
        Fixes:
          1. motion_samples was empty → _analyse_motion always "stationary"
          2. all_seconds with epoch values → normalise to offset-from-first
          3. Quadrant fallback overridden by real displacement data
        """
        from app.storage.models import TrackEvent, DetectedObject
        from app.detection.temporal_analyzer import TemporalAnalyzer
        from app.detection.event_generator import TrackState, MotionSample
        from sqlalchemy.orm.attributes import flag_modified

        entry_events = (
            db.query(TrackEvent)
            .filter(TrackEvent.video_filename == key,
                    TrackEvent.event_type == "entry")
            .all()
        )
        if not entry_events:
            return

        all_dets = (
            db.query(DetectedObject)
            .filter(DetectedObject.video_filename == key)
            .order_by(DetectedObject.frame_second_offset)
            .all()
        )

        det_by_track = {}
        samples_by_track = {}
        seconds_by_track = {}

        for d in all_dets:
            if d.track_id is None:
                continue
            det_by_track.setdefault(d.track_id, []).append(d)
            seconds_by_track.setdefault(d.track_id, []).append(d.frame_second_offset)
            cx = (d.bbox_x1 + d.bbox_x2) / 2.0
            cy = (d.bbox_y1 + d.bbox_y2) / 2.0
            w  = abs(d.bbox_x2 - d.bbox_x1)
            h  = abs(d.bbox_y2 - d.bbox_y1)
            samples_by_track.setdefault(d.track_id, []).append(
                MotionSample(second=d.frame_second_offset, cx=cx, cy=cy, w=w, h=h))

        track_states = {}
        for ev in entry_events:
            epoch_secs   = sorted(seconds_by_track.get(ev.track_id) or
                                  [ev.first_seen_second, ev.last_seen_second])
            motion_samps = sorted(samples_by_track.get(ev.track_id, []),
                                  key=lambda m: m.second)
            t0           = epoch_secs[0]
            rel_secs     = [s - t0 for s in epoch_secs]
            rel_samples  = [MotionSample(second=m.second - t0,
                                         cx=m.cx, cy=m.cy, w=m.w, h=m.h)
                            for m in motion_samps]
            duration = ev.last_seen_second - ev.first_seen_second
            # For live tracks: if duration is 0 (exit not yet flushed),
            # try to compute from wall-clock attributes
            if duration <= 0 and ev.attributes:
                entry_wall = ev.attributes.get("entry_wall_time")
                exit_wall  = ev.attributes.get("exit_wall_time")
                if entry_wall and exit_wall:
                    import datetime as _dt
                    try:
                        _tw_in  = _dt.datetime.fromisoformat(entry_wall)
                        _tw_out = _dt.datetime.fromisoformat(exit_wall)
                        duration = max(1.0, (_tw_out - _tw_in).total_seconds())
                    except Exception:
                        duration = max(1.0, len(seconds_by_track.get(ev.track_id, [1, 2])))
                else:
                    duration = max(1.0, len(seconds_by_track.get(ev.track_id, [1, 2])))

            track_states[ev.track_id] = TrackState(
                track_id=ev.track_id,
                object_class=ev.object_class,
                first_seen=0.0,
                last_seen=round(duration, 1),
                frame_count=max(2, len(rel_secs)),
                best_second=0.0,
                best_confidence=ev.best_confidence or 0.5,
                best_crop_path=ev.best_crop_path,
                all_seconds=rel_secs,
                motion_samples=rel_samples,
            )

        # Guard: remove zero-duration tracks that TemporalAnalyzer can't process
        track_states = {
            tid: ts for tid, ts in track_states.items()
            if ts.last_seen > 0 or len(ts.all_seconds) > 1
        }
        if not track_states:
            self.logger.warning("window_temporal_no_valid_tracks", key=key)
            return

        try:
            behaviours = TemporalAnalyzer().analyze(track_states, det_by_track)
        except Exception as te:
            self.logger.warning("window_temporal_analyze_failed", key=key, error=str(te))
            return
        beh_map    = {b.track_id: b for b in behaviours}

        all_events = db.query(TrackEvent).filter(
            TrackEvent.video_filename == key).all()
        for ev in all_events:
            beh = beh_map.get(ev.track_id)
            if beh:
                attrs = dict(ev.attributes or {})
                attrs["temporal"] = beh.to_dict()
                ev.attributes = attrs
                flag_modified(ev, "attributes")
        db.commit()
        self.logger.info("window_temporal_written", key=key, tracks=len(behaviours))

    def _run_reembed(self, key: str, db):
        """Re-embed all TrackEvent rag_text after attribute extraction rewrites it."""
        try:
            from app.storage.models import TrackEvent, TrackEventEmbedding
            from app.rag.embedder import OllamaEmbedder
            embedder = OllamaEmbedder()
            events = db.query(TrackEvent).filter(
                TrackEvent.video_filename == key,
                TrackEvent.rag_text.isnot(None),
            ).all()
            count = 0
            for ev in events:
                try:
                    vec = embedder.embed(ev.rag_text)
                    if vec is None:
                        continue
                    emb = db.query(TrackEventEmbedding).filter(
                        TrackEventEmbedding.track_event_id == ev.id).first()
                    if emb:
                        emb.embedding = vec
                    else:
                        from app.core.config import get_settings as _gs
                        _s = _gs()
                        db.add(TrackEventEmbedding(
                            track_event_id=ev.id,
                            embedding=vec,
                            model_name=_s.embed_model,
                        ))
                    count += 1
                except Exception:
                    pass
            db.commit()
            self.logger.info("window_reembed_done", key=key, count=count)
        except Exception as e:
            self.logger.warning("window_reembed_failed", key=key, error=str(e))

    # ── DB helpers ────────────────────────────────────────────────────────────

    def _create_processing_status(self, key: str):
        from app.storage.database import SessionLocal
        from app.storage.models import ProcessingStatus
        db = SessionLocal()
        try:
            existing = db.query(ProcessingStatus).filter(
                ProcessingStatus.video_filename == key).first()
            if not existing:
                ps = ProcessingStatus(
                    video_filename=key,
                    camera_id=self.settings.camera_id,
                    status="live",
                    started_at=datetime.utcnow(),
                )
                db.add(ps)
                db.commit()
        except Exception as e:
            self.logger.debug("window_status_create_failed", key=key, error=str(e))
            try: db.rollback()
            except Exception: pass
        finally:
            db.close()

    def _mark_status(self, key: str, status: str):
        from app.storage.database import SessionLocal
        from app.storage.models import ProcessingStatus
        db = SessionLocal()
        try:
            ps = db.query(ProcessingStatus).filter(
                ProcessingStatus.video_filename == key).first()
            if ps:
                ps.status = status
                if status == "completed":
                    ps.completed_at = datetime.utcnow()
                db.commit()
        except Exception as e:
            self.logger.debug("window_status_update_failed",
                              key=key, status=status, error=str(e))
            try: db.rollback()
            except Exception: pass
        finally:
            db.close()