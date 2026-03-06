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
        """Stop the rotation loop. Closes current window."""
        self._stop_evt.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=15)
        with self._lock:
            if self._current_key:
                self._close_window(self._current_key)
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
        """
        t = threading.Thread(
            target=self._run_postprocess,
            args=(key,),
            daemon=True,
            name=f"postprocess_{key}",
        )
        t.start()

    def _run_postprocess(self, key: str):
        """
        Sequential post-processing for a closed window.
        Runs: attribute extraction → temporal analysis → summary generation.
        All existing pipeline functions are reused unchanged.
        """
        from app.storage.database import SessionLocal
        db = SessionLocal()
        try:
            self.logger.info("window_postprocess_start", key=key)

            # 1. Attribute extraction (Phase 6B)
            try:
                from app.detection.attribute_processor import AttributeProcessor
                proc = AttributeProcessor(db)
                proc.run(key)
                self.logger.info("window_attrs_done", key=key)
            except Exception as e:
                self.logger.warning("window_attrs_failed", key=key, error=str(e))

            # 2. Temporal analysis
            try:
                from app.detection.temporal_analyzer import TemporalAnalyzer
                ta = TemporalAnalyzer(db)
                ta.run(key)
                self.logger.info("window_temporal_done", key=key)
            except Exception as e:
                self.logger.warning("window_temporal_failed", key=key, error=str(e))

            # 3. Memory graph
            try:
                from app.storage.memory_graph import MemoryGraphBuilder
                mgb = MemoryGraphBuilder(db)
                mgb.build(key)
                self.logger.info("window_memory_graph_done", key=key)
            except Exception as e:
                self.logger.warning("window_memory_graph_failed", key=key, error=str(e))

            # 4. Activity captions (minicpm-v one-liner per track)
            try:
                from app.detection.activity_detector import run_activity_captions_for_window
                n = run_activity_captions_for_window(key, db)
                self.logger.info("window_activity_captions_done", key=key, captioned=n)
            except Exception as e:
                self.logger.warning("window_activity_captions_failed", key=key, error=str(e))

            # 5. Summary generation
            try:
                from app.rag.summarizer import VideoSummarizer
                summarizer = VideoSummarizer(db)
                summarizer.summarize(key)
                self.logger.info("window_summary_done", key=key)
            except Exception as e:
                self.logger.warning("window_summary_failed", key=key, error=str(e))

            self._mark_status(key, "completed")
            self.logger.info("window_postprocess_complete", key=key)

        except Exception as e:
            self.logger.error("window_postprocess_error", key=key, error=str(e))
            self._mark_status(key, "failed")
        finally:
            db.close()

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