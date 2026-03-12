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

import heapq
import itertools
import threading
import time
from datetime import datetime, timedelta
from typing import Optional

from app.core.config import get_settings
from app.core.logging import get_logger


class OllamaLaneBusyError(RuntimeError):
    """Raised when the Ask lane cannot acquire the Ollama semaphore within the timeout."""


class _PriorityLane:
    """
    Drop-in replacement for threading.Semaphore with priority support.
    Lower priority number = served first (0 = highest priority).

    When multiple threads are waiting and a slot is released, the thread
    with the lowest priority number receives it. Ties are broken in FIFO order.
    """

    def __init__(self, value: int = 1):
        self._mutex = threading.Lock()
        self._value = value
        self._waiters: list = []  # min-heap: (priority, seq, Event)
        self._seq_gen = itertools.count()

    def acquire(self, priority: int = 10, blocking: bool = True,
                timeout: Optional[float] = None) -> bool:
        deadline = (time.monotonic() + timeout) if timeout is not None else None
        with self._mutex:
            if self._value > 0:
                self._value -= 1
                return True
            if not blocking:
                return False
            evt = threading.Event()
            entry = (priority, next(self._seq_gen), evt)
            heapq.heappush(self._waiters, entry)

        wait_secs = None if deadline is None else max(0.0, deadline - time.monotonic())
        acquired = evt.wait(timeout=wait_secs)
        if acquired:
            return True
        # Timed out — remove our entry from the queue
        with self._mutex:
            try:
                self._waiters.remove(entry)
                heapq.heapify(self._waiters)
            except ValueError:
                # release() granted us the slot just as we timed out.
                # We can't use it, so pass it on to the next waiter.
                self._release_next()
        return False

    def release(self):
        with self._mutex:
            self._release_next()

    def _release_next(self):
        """Must be called with self._mutex held."""
        if self._waiters:
            _, _, evt = heapq.heappop(self._waiters)
            evt.set()  # hand slot directly to highest-priority waiter
        else:
            self._value += 1


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
        self._queued_keys    = set()

        # Priority-aware lane shared across Ask + pipeline Ollama generate calls.
        # Ask queries get priority=0 (served first), pipeline steps get priority=10.
        # This ensures user Ask queries jump ahead of background post-processing
        # and are served on the very next slot release.
        self._ollama_sem = _PriorityLane(
            max(1, self.settings.ollama_concurrency, self.settings.ask_ollama_concurrency, self.settings.pipeline_ollama_concurrency)
        )

    # ── Public API ────────────────────────────────────────────────────────────

    def start(self):
        """Start the window rotation thread."""
        self._open_window(datetime.utcnow())
        self._stop_evt.clear()
        self._thread = threading.Thread(
            target=self._rotation_loop, daemon=True, name="window_manager")
        self._thread.start()
        self._resume_stuck_windows()
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
        Spawn a background thread to run attribute extraction ??? temporal ??? summary.
        Non-blocking: stream continues while this runs.
        All Ollama generate calls share one bounded lane.
        """
        with self._lock:
            if key in self._queued_keys:
                return
            self._queued_keys.add(key)
        t = threading.Thread(
            target=self._run_postprocess,
            args=(key,),
            daemon=True,
            name=f"postprocess_{key}",
        )
        t.start()
    def _resume_stuck_windows(self):
        """Queue windows that were left in post_processing after a restart."""
        try:
            from app.storage.database import SessionLocal
            from app.storage.models import ProcessingStatus
            db = SessionLocal()
            try:
                rows = db.query(ProcessingStatus).filter(
                    ProcessingStatus.status == "post_processing"
                ).all()
                for row in rows:
                    self.logger.info("window_resume_queued", key=row.video_filename)
                    self._queue_postprocess(row.video_filename)
            finally:
                db.close()
        except Exception as exc:
            self.logger.warning("window_resume_scan_failed", error=str(exc))

    def _step_completed(self, db, key: str, attr_name: str) -> bool:
        from app.storage.models import ProcessingStatus
        ps = db.query(ProcessingStatus).filter(ProcessingStatus.video_filename == key).first()
        return bool(getattr(ps, attr_name, False)) if ps else False

    def _mark_step_completed(self, db, key: str, attr_name: str):
        from app.storage.models import ProcessingStatus
        ps = db.query(ProcessingStatus).filter(ProcessingStatus.video_filename == key).first()
        if ps:
            setattr(ps, attr_name, True)
            db.commit()

    # Priority constants for the Ollama lane.
    # Lower number = higher priority = served first when slot is released.
    PRIORITY_ASK      = 0   # Interactive user queries — highest priority
    PRIORITY_USER_API = 5   # User-triggered API calls (manual summary, etc.)
    PRIORITY_PIPELINE  = 10  # Background post-processing steps

    def ollama_ctx(self, lane: str, key: Optional[str] = None,
                   timeout: Optional[float] = None, priority: int = PRIORITY_USER_API):
        import contextlib

        @contextlib.contextmanager
        def _ctx():
            wait_start = time.monotonic()
            self.logger.info("ollama_lane_waiting", lane=lane, key=key, priority=priority)
            if timeout is not None:
                acquired = self._ollama_sem.acquire(
                    priority=priority, blocking=True, timeout=timeout)
            else:
                acquired = self._ollama_sem.acquire(
                    priority=priority, blocking=True)
            waited_s = round(time.monotonic() - wait_start, 3)
            if not acquired:
                self.logger.warning(
                    "ollama_lane_timeout",
                    lane=lane, key=key, waited_s=waited_s, priority=priority,
                )
                raise OllamaLaneBusyError(
                    f"Ollama is busy processing video pipeline tasks. "
                    f"Waited {waited_s:.0f}s. Please try again in a few minutes."
                )
            self.logger.info("ollama_lane_acquired", lane=lane, key=key,
                             waited_s=waited_s, priority=priority)
            try:
                yield {"waited_s": waited_s}
            finally:
                self._ollama_sem.release()
                self.logger.info("ollama_lane_released", lane=lane, key=key)

        return _ctx()

    def ask_ollama_ctx(self):
        # Ask lane: highest priority (0) so it jumps ahead of any queued pipeline steps.
        # Uses a timeout so it never blocks indefinitely.
        ask_timeout = float(getattr(self.settings, "ask_lane_timeout_seconds", 60))
        return self.ollama_ctx("ask", timeout=ask_timeout, priority=self.PRIORITY_ASK)

    def _pipeline_ollama_ctx(self, key: str, step_name: str):
        # Pipeline: lowest priority (10) — yields to Ask queries.
        return self.ollama_ctx(f"pipeline_{step_name}", key=key, priority=self.PRIORITY_PIPELINE)

    def pipeline_yield_point(self, key: str, step_name: str):
        """
        Call between individual Ollama operations inside a pipeline step.
        If a higher-priority waiter (Ask) is queued, temporarily releases
        the semaphore so Ask can run, then re-acquires for the pipeline.

        Must be called while the pipeline thread holds the semaphore.
        """
        has_higher = False
        with self._ollama_sem._mutex:
            for p, _, _ in self._ollama_sem._waiters:
                if p < self.PRIORITY_PIPELINE:
                    has_higher = True
                    break
        if has_higher:
            self.logger.info("pipeline_yielding_for_ask", key=key, step=step_name)
            self._ollama_sem.release()
            # Brief sleep so the Ask thread can grab the slot
            time.sleep(0.2)
            # Re-acquire at pipeline priority (blocks until Ask finishes)
            self._ollama_sem.acquire(priority=self.PRIORITY_PIPELINE, blocking=True)
            self.logger.info("pipeline_resumed_after_yield", key=key, step=step_name)

    def _make_yield_cb(self, key: str, step_name: str):
        """Create a yield callback for passing to processors."""
        def _cb():
            self.pipeline_yield_point(key, step_name)
        return _cb

    def _log_step_result(self, key: str, step_name: str, started_at: float, status: str, **meta):
        elapsed_s = round(time.monotonic() - started_at, 1)
        self.logger.info(
            "pipeline_step_result",
            key=key,
            step=step_name,
            status=status,
            elapsed_s=elapsed_s,
            **meta,
        )

    def _run_postprocess(self, key: str):
        """
        Sequential post-processing for a closed window.
        Pipeline: attributes -> temporal -> timeline -> memory -> activity -> reembed -> summary
        Each step updates ProcessingStatus.current_step so the UI can show live progress.
        """
        import time
        from app.storage.database import SessionLocal

        db = SessionLocal()
        _t0 = time.monotonic()

        def _elapsed():
            return round(time.monotonic() - _t0, 1)

        def _step_failure(step_name: str, started_at: float, exc: Exception):
            self._log_step_result(
                key,
                step_name,
                started_at,
                "failed",
                total_s=_elapsed(),
                error=str(exc),
                error_type=exc.__class__.__name__,
                timeout=bool(isinstance(exc, TimeoutError)),
            )

        try:
            time.sleep(3)
            self.logger.info("window_postprocess_start", key=key)

            if not self._step_completed(db, key, "attrs_completed"):
                self._set_step(key, "Attributes", db)
                step_started = time.monotonic()
                try:
                    from app.core.config import get_settings as _gs_attr

                    settings = _gs_attr()
                    if settings.enable_phase_6b and settings.attribute_policy not in ("off", "manual_only"):
                        self.logger.info("window_attrs_waiting_sem", key=key)
                        with self._pipeline_ollama_ctx(key, "attrs"):
                            self.logger.info("window_attrs_sem_acquired", key=key)
                            from app.detection.attribute_processor import AttributeProcessor

                            n_attr = AttributeProcessor(db).run(
                                key, yield_cb=self._make_yield_cb(key, "attrs"))
                        self._mark_step_completed(db, key, "attrs_completed")
                        self._log_step_result(
                            key,
                            "attrs",
                            step_started,
                            "completed",
                            total_s=_elapsed(),
                            tracks=n_attr,
                        )
                    else:
                        self._mark_step_completed(db, key, "attrs_completed")
                        self._log_step_result(
                            key,
                            "attrs",
                            step_started,
                            "skipped",
                            total_s=_elapsed(),
                            reason="policy_or_phase_disabled",
                        )
                except Exception as exc:
                    _step_failure("attrs", step_started, exc)
                    self.logger.warning("window_attrs_failed", key=key, error=str(exc))

            self._set_step(key, "Temporal", db)
            step_started = time.monotonic()
            try:
                self._run_temporal(key, db)
                self._log_step_result(key, "temporal", step_started, "completed", total_s=_elapsed())
            except Exception as exc:
                _step_failure("temporal", step_started, exc)
                self.logger.warning("window_temporal_failed", key=key, error=str(exc))

            self._set_step(key, "Timeline", db)
            step_started = time.monotonic()
            try:
                from app.core.config import get_settings
                from app.detection.timeline_builder import TimelineBuilder

                TimelineBuilder(db).build(key, get_settings().camera_id)
                self._log_step_result(key, "timeline", step_started, "completed", total_s=_elapsed())
            except Exception as exc:
                _step_failure("timeline", step_started, exc)
                self.logger.warning("window_timeline_failed", key=key, error=str(exc))

            self._set_step(key, "Memory", db)
            step_started = time.monotonic()
            try:
                from app.core.config import get_settings
                from app.storage.memory_graph import MemoryGraphBuilder

                MemoryGraphBuilder(db).build(key, get_settings().camera_id)
                self._log_step_result(key, "memory", step_started, "completed", total_s=_elapsed())
            except Exception as exc:
                _step_failure("memory", step_started, exc)
                self.logger.warning("window_memory_graph_failed", key=key, error=str(exc))

            if not self._step_completed(db, key, "activity_completed"):
                self._set_step(key, "Activity AI", db)
                step_started = time.monotonic()
                try:
                    from app.core.config import get_settings as _gs_act

                    settings = _gs_act()
                    if settings.enable_activity_captions and settings.multimodal_model:
                        self.logger.info("window_activity_waiting_sem", key=key)
                        with self._pipeline_ollama_ctx(key, "activity"):
                            self.logger.info("window_activity_sem_acquired", key=key)
                            from app.detection.activity_detector import run_activity_captions_for_window

                            n_cap = run_activity_captions_for_window(
                                key, db, yield_cb=self._make_yield_cb(key, "activity"))
                        self._mark_step_completed(db, key, "activity_completed")
                        self._log_step_result(
                            key,
                            "activity",
                            step_started,
                            "completed",
                            total_s=_elapsed(),
                            captioned=n_cap,
                        )
                    else:
                        self._mark_step_completed(db, key, "activity_completed")
                        self._log_step_result(
                            key,
                            "activity",
                            step_started,
                            "skipped",
                            total_s=_elapsed(),
                            reason="disabled",
                        )
                except Exception as exc:
                    _step_failure("activity", step_started, exc)
                    self.logger.warning("window_activity_captions_failed", key=key, error=str(exc))

            if not self._step_completed(db, key, "reembed_completed"):
                self._set_step(key, "Embeddings", db)
                step_started = time.monotonic()
                try:
                    count = self._run_reembed(key, db)
                    self._mark_step_completed(db, key, "reembed_completed")
                    self._log_step_result(
                        key,
                        "reembed",
                        step_started,
                        "completed",
                        total_s=_elapsed(),
                        embeddings=count,
                    )
                except Exception as exc:
                    _step_failure("reembed", step_started, exc)
                    self.logger.warning("window_reembed_failed", key=key, error=str(exc))

            if not self._step_completed(db, key, "summary_completed"):
                self._set_step(key, "Summary", db)
                step_started = time.monotonic()
                try:
                    from app.core.config import get_settings as _gs_sum
                    if not _gs_sum().auto_summarize:
                        # AUTO_SUMMARIZE=false — skip now, user triggers via
                        # POST /api/v1/summarize/{video} or the Summary tab.
                        self._mark_step_completed(db, key, "summary_completed")
                        self._log_step_result(
                            key, "summary", step_started, "skipped",
                            total_s=_elapsed(), reason="auto_summarize_disabled",
                        )
                    else:
                        self.logger.info("window_summary_waiting_sem", key=key)
                        with self._pipeline_ollama_ctx(key, "summary"):
                            self.logger.info("window_summary_sem_acquired", key=key)
                            from app.rag.summarizer import VideoSummarizer

                            summary = VideoSummarizer(db).summarize_from_tracks(key, force=True)
                        self._mark_step_completed(db, key, "summary_completed")
                        self._log_step_result(
                            key,
                            "summary",
                            step_started,
                            "completed",
                            total_s=_elapsed(),
                            summary_chars=len(summary.summary_text or ""),
                            summary_model=summary.model_name,
                        )
                except Exception as exc:
                    _step_failure("summary", step_started, exc)
                    self.logger.warning("window_summary_failed", key=key, error=str(exc))

            try:
                from app.storage.models import DetectedObject, ProcessingStatus, TrackEvent

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
                    ps.scenes_detected = n_tracks
                    ps.scenes_captioned = n_dets
                    attempted = db.query(TrackEvent).filter(
                        TrackEvent.video_filename == key,
                        TrackEvent.event_type == "entry",
                        TrackEvent.attributes.isnot(None),
                    ).all()
                    attempted_count = sum(1 for ev in attempted if (ev.attributes or {}).get("attr_attempted"))
                    ps.phase_6b_completed = bool(ps.attrs_completed)
                    ps.phase_6b_tracks_attributed = attempted_count
                    ps.current_step = None
                    db.commit()
            except Exception as exc:
                self.logger.warning("window_status_counts_failed", key=key, error=str(exc))

            self._mark_status(key, "completed")
            self.logger.info("window_postprocess_complete", key=key, total_s=_elapsed())

        except Exception as exc:
            self.logger.error("window_postprocess_error", key=key, error=str(exc))
            self._mark_status(key, "failed")
        finally:
            with self._lock:
                self._queued_keys.discard(key)
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
        from app.rag.embedder import OllamaEmbedder
        from app.storage.models import TrackEvent, TrackEventEmbedding

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
        return count

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