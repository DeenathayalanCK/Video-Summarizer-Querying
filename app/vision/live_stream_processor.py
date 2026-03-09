"""
live_stream_processor.py — Continuous RTSP stream processing.

Key design: live detections write into TrackEvent + DetectedObject using the
current window key (e.g. gate_cam_01_20260305_1600) as video_filename.
This means ALL existing tabs (Ask, Search, Timeline, Temporal, Summary,
Detections) work against live data with zero code changes.

Cross-boundary track handling:
  When the window key changes mid-track, the active LiveTrack is "split":
  - The old window gets an exit event written with last_seen = window boundary
  - The new window gets an entry event written with first_seen = window boundary
  Both windows contain the person, satisfying the "both windows" requirement.

Person identity (PersonSession / PersonIdentity) is separate from the window
system — identity persists across all windows for the same physical person.
"""

import os
import cv2
import threading
import time
import numpy as np
from datetime import datetime
from typing import Optional, Dict
from dataclasses import dataclass, field

from app.core.config import get_settings
from app.core.logging import get_logger
from app.detection.detector import ObjectDetector, Detection
from app.detection.appearance_reid import AppearanceReID
from app.detection.crop_utils import extract_crop


# ── Per-track in-memory state ─────────────────────────────────────────────────

@dataclass
class LiveTrack:
    track_id:        int
    person_label:    str
    session_id:      str
    first_seen:      datetime
    last_seen:       datetime
    window_key:      str           # which window this track started in
    best_confidence: float = 0.0
    best_crop_path:  Optional[str] = None
    embedding:       Optional[np.ndarray] = None
    last_state:      str = "unknown"
    last_bbox:       tuple = (0, 0, 0, 0)
    frame_count:     int = 0
    # Activity detection (Stage 1 — YOLO objects on crop)
    objects_nearby:  list = field(default_factory=list)
    activity_hint:   str = "present"
    # Face detection results (populated at track close)
    face_crop_path:  Optional[str] = None
    face_detected:   bool = False
    face_confidence: float = 0.0
    # For TrackEvent writing
    track_event_written: bool = False   # entry event written to DB
    det_obj_id:      Optional[str] = None  # DetectedObject UUID


# ── Annotation helpers ────────────────────────────────────────────────────────

_COLORS = [
    (0, 165, 255), (0, 255, 0), (255, 0, 255), (0, 255, 255),
    (255, 128, 0), (128, 0, 255), (0, 128, 255), (255, 0, 128),
    (255, 200, 0), (0, 200, 255),
]

def _color_for(label: str) -> tuple:
    return _COLORS[sum(ord(c) for c in label) % len(_COLORS)]

def _classify_state(bbox_now, bbox_prev, frame_count: int) -> str:
    if bbox_prev is None or frame_count < 3:
        return "entering"
    cx_n = (bbox_now[0]  + bbox_now[2]) / 2
    cy_n = (bbox_now[1]  + bbox_now[3]) / 2
    cx_p = (bbox_prev[0] + bbox_prev[2]) / 2
    cy_p = (bbox_prev[1] + bbox_prev[3]) / 2
    dist  = ((cx_n - cx_p)**2 + (cy_n - cy_p)**2) ** 0.5
    box_h = bbox_now[3] - bbox_now[1]
    if box_h == 0: return "stationary"
    ratio = dist / box_h
    if ratio > 0.30: return "running"
    if ratio > 0.07: return "walking"
    return "stationary"


# ── Thread-safe frame buffer ──────────────────────────────────────────────────

class FrameBuffer:
    def __init__(self):
        self._lock  = threading.Lock()
        self._frame: Optional[bytes] = None
        self._ts:    float = 0.0

    def put(self, jpeg: bytes):
        with self._lock:
            self._frame = jpeg
            self._ts    = time.time()

    def get(self) -> Optional[bytes]:
        with self._lock:
            return self._frame

    def age(self) -> float:
        return time.time() - self._ts if self._ts else 999.0


# ── Main processor ────────────────────────────────────────────────────────────

class LiveStreamProcessor:
    _instance: Optional["LiveStreamProcessor"] = None
    _cls_lock  = threading.Lock()

    @classmethod
    def get_instance(cls) -> "LiveStreamProcessor":
        with cls._cls_lock:
            if cls._instance is None:
                cls._instance = cls()
            return cls._instance

    def __init__(self):
        self.settings     = get_settings()
        self.logger       = get_logger()
        self.detector     = ObjectDetector()
        self.reid         = AppearanceReID()
        from app.detection.activity_detector import ActivityDetector
        self.activity      = ActivityDetector()
        from app.detection.face_detector import FaceDetector
        self.face_detector = FaceDetector.get_instance()

        self._thread:     Optional[threading.Thread] = None
        self._stop_evt    = threading.Event()
        self._running     = False
        self.frame_buffer = FrameBuffer()

        self._tracks:      Dict[int, LiveTrack] = {}
        self._tracks_lock  = threading.Lock()
        self._next_label_num  = 1
        self._label_lock      = threading.Lock()
        # Labels reserved mid-frame (assigned but track not yet in self._tracks)
        # Prevents two detections in the same batch both getting "P1"
        self._reserved_labels: set = set()

        # Recently-lost tracks indexed by position for spatial re-identification.
        # Key: person_label, Value: {cx, cy, box_h, last_seen, session_id, embedding}
        # Kept for MERGE_GAP_SECONDS so stationary desk workers re-acquire same label.
        self._lost_tracks: dict = {}
        self._lost_tracks_lock = threading.Lock()

        # Bug 4 fix: per-track_id wall-time of last activity detection poll.
        # Activity detection runs every N frames OR every 5 real-seconds,
        # whichever comes first — not just on crop-improvement events.
        self._last_activity_check: Dict[int, float] = {}   # track_id → epoch float

        # Bug 2 fix: per-track_id wall-time of last activity_snapshot write.
        # write_activity_snapshot() is called at most once per 60s per track.
        self._last_snapshot_time: Dict[int, datetime] = {}  # track_id → datetime

    # ── Control ───────────────────────────────────────────────────────────────

    def start(self) -> bool:
        if self._running:
            return False
        if not self.settings.video_input_path:
            self.logger.error("live_start_failed", reason="VIDEO_INPUT_PATH not set")
            return False
        self._stop_evt.clear()
        self._thread = threading.Thread(
            target=self._run_loop, daemon=True, name="live_stream")
        self._thread.start()
        self._running = True
        self.logger.info("live_stream_started", url=self.settings.video_input_path)
        return True

    def stop(self):
        self._stop_evt.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=10)
        self._running = False

    @property
    def is_running(self) -> bool:
        return self._running and self._thread is not None and self._thread.is_alive()

    def get_frame(self) -> Optional[bytes]:
        return self.frame_buffer.get()

    def get_active_tracks(self) -> list:
        with self._tracks_lock:
            now     = datetime.utcnow()
            timeout = self.settings.live_exit_timeout_seconds
            result  = []
            for t in self._tracks.values():
                age = (now - t.last_seen).total_seconds()
                if age <= timeout * 2:
                    result.append({
                        "track_id":       t.track_id,
                        "person_label":   t.person_label,
                        "session_id":     t.session_id,
                        "entry_time":     t.first_seen.isoformat(),
                        "last_seen":      t.last_seen.isoformat(),
                        "last_state":     t.last_state,
                        "best_crop_path": t.best_crop_path,
                        "active":         age <= timeout,
                        "duration_seconds": (t.last_seen - t.first_seen).total_seconds(),
                        # Activity detection results (populated every 5 frames)
                        "objects_nearby": list(t.objects_nearby) if hasattr(t, "objects_nearby") else [],
                        "activity_hint":  t.activity_hint if hasattr(t, "activity_hint") else "present",
                        "face_detected":   getattr(t, "face_detected", False),
                        "face_crop_path":  getattr(t, "face_crop_path", None),
                        "face_confidence": getattr(t, "face_confidence", 0.0),
                    })
            return sorted(result, key=lambda x: x["person_label"])

    # ── Main loop ─────────────────────────────────────────────────────────────

    def _run_loop(self):
        from app.storage.database import SessionLocal
        from app.vision.window_manager import WindowManager

        wm       = WindowManager.get_instance()
        read_db  = SessionLocal()
        quality  = self.settings.live_stream_jpeg_quality
        sample_interval = 1.0 / max(1, self.settings.live_sample_fps)

        try:
            self._init_label_counter(read_db)
        except Exception as e:
            self.logger.warning("label_init_failed", error=str(e))

        while not self._stop_evt.is_set():
            cap = None
            try:
                self.logger.info("live_connecting", url=self.settings.video_input_path)
                cap = cv2.VideoCapture(self.settings.video_input_path)
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)
                if not cap.isOpened():
                    raise RuntimeError("Cannot open RTSP stream")
                self.logger.info("live_connected")

                last_sample = 0.0
                while not self._stop_evt.is_set():
                    ret, frame = cap.read()
                    if not ret:
                        break

                    # Always push annotated frame to MJPEG buffer
                    annotated = self._annotate_frame(frame)
                    ok, buf = cv2.imencode(
                        ".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, quality])
                    if ok:
                        self.frame_buffer.put(bytes(buf))

                    now = time.time()
                    if now - last_sample < sample_interval:
                        continue
                    last_sample = now
                    wall_time   = datetime.utcnow()
                    window_key  = wm.current_key or wm.key_for_time(wall_time)

                    # Detect
                    try:
                        dets = self.detector.detect_with_tracking(frame, now)
                    except Exception as e:
                        self.logger.warning("detect_error", error=str(e))
                        continue

                    write_db = SessionLocal()
                    try:
                        if dets and dets.has_objects:
                            self._update_tracks(
                                frame, dets, wall_time, window_key,
                                write_db, read_db)
                        self._close_stale_tracks(wall_time, window_key, write_db)
                        # Check for window boundary crossing on active tracks
                        self._handle_window_boundary(wall_time, window_key, write_db)
                    except Exception as e:
                        self.logger.error("track_update_error", error=str(e))
                        try: write_db.rollback()
                        except Exception: pass
                    finally:
                        write_db.close()

            except Exception as e:
                self.logger.error("live_loop_error", error=str(e))
            finally:
                if cap: cap.release()

            if not self._stop_evt.is_set():
                self._stop_evt.wait(timeout=5)

        # Graceful shutdown
        shutdown_db = SessionLocal()
        try:
            now = datetime.utcnow()
            wk  = wm.current_key or wm.key_for_time(now)
            self._close_all_sessions(now, wk, shutdown_db)
        except Exception: pass
        finally:
            shutdown_db.close()
            read_db.close()
            self._running = False

    # ── Track management ──────────────────────────────────────────────────────

    def _update_tracks(self, frame, dets, wall_time, window_key,
                       write_db, read_db):
        for det in dets.detections:
            if det.object_class != "person" or det.track_id is None:
                continue
            bbox_now = (det.x1, det.y1, det.x2, det.y2)

            with self._tracks_lock:
                existing = self._tracks.get(det.track_id)

            if existing:
                prev_bbox = existing.last_bbox
                state     = _classify_state(bbox_now, prev_bbox, existing.frame_count)
                with self._tracks_lock:
                    existing.last_seen    = wall_time
                    existing.last_state   = state
                    existing.last_bbox    = bbox_now
                    existing.frame_count += 1

                if det.confidence > (existing.best_confidence or 0) + 0.08:
                    crop = extract_crop(frame, det)
                    if crop is not None and crop.size > 0:
                        path = self._save_crop(crop, existing.person_label, wall_time)
                        if path:
                            # Stage 1 activity: detect objects in context crop (on crop improvement)
                            objs, hint = self.activity.detect_on_context(frame, bbox_now)
                            with self._tracks_lock:
                                existing.best_crop_path  = path
                                existing.best_confidence = det.confidence
                                existing.objects_nearby  = objs
                                existing.activity_hint   = hint
                            self._last_activity_check[det.track_id] = time.time()
                            self._update_identity_embedding(
                                existing.person_label, path, write_db)

                # Bug 4 fix: also poll activity every 5 real-seconds regardless of
                # crop improvement — a seated desk worker's confidence never improves
                # after the first few seconds, so without this the activity freezes.
                _now_f = time.time()
                _last_act = self._last_activity_check.get(det.track_id, 0.0)
                if _now_f - _last_act >= 5.0:
                    objs_p, hint_p = self.activity.detect_on_context(frame, bbox_now)
                    with self._tracks_lock:
                        existing.objects_nearby = objs_p
                        existing.activity_hint  = hint_p
                    self._last_activity_check[det.track_id] = _now_f

                # Bug 2 fix: write an activity_snapshot every 60s for notable activities
                _last_snap = self._last_snapshot_time.get(det.track_id)
                _snap_due  = (
                    _last_snap is None or
                    (wall_time - _last_snap).total_seconds() >= 60
                )
                if _snap_due and existing.activity_hint not in ("present", None, ""):
                    from app.detection.activity_detector import write_activity_snapshot
                    write_activity_snapshot(existing, window_key, wall_time, write_db)
                    self._last_snapshot_time[det.track_id] = wall_time

                if existing.frame_count % 10 == 0:
                    self._persist_session(existing, write_db)
                    self._update_track_event(existing, write_db)

            else:
                # New ByteTrack ID
                crop      = extract_crop(frame, det)
                crop_path = None
                embedding = None
                if crop is not None and crop.size > 0:
                    crop_path = self._save_crop(
                        crop, f"tmp{det.track_id}", wall_time)
                    if crop_path:
                        embedding = self.reid.embed_crop(crop_path)
                # Initial activity detection on context crop
                _init_objects, _init_hint = (
                    self.activity.detect_on_context(frame, bbox_now)
                    if crop is not None and crop.size > 0
                    else ([], "present")
                )

                # ── Step 1: Position-first match (best for static/desk scenes) ──
                # Works regardless of embedding quality. A desk worker who
                # ByteTrack loses and re-acquires stays within ~80px of their seat.
                reused_label, reused_session = self._match_by_position(
                    bbox_now, wall_time,
                    merge_gap_seconds=self.settings.live_exit_timeout_seconds * 4)

                # ── Step 2: Appearance match (moving persons, new arrivals) ──
                if not reused_label and embedding is not None:
                    reused_label, reused_session = self._match_active_track(
                        embedding, det.track_id, bbox_now)

                if reused_label:
                    # Also clean position cache if this label was there
                    with self._lost_tracks_lock:
                        self._lost_tracks.pop(reused_label, None)
                    # Fix: rename tmp crop to labelled name for reused-label tracks
                    if crop_path and f"tmp{det.track_id}" in crop_path:
                        _rl_new = crop_path.replace(f"tmp{det.track_id}", reused_label)
                        try:
                            os.rename(crop_path, _rl_new)
                            crop_path = _rl_new
                            self._update_identity_crop_path(
                                reused_label, crop_path, write_db)
                        except Exception:
                            pass
                    track = LiveTrack(
                        track_id=det.track_id,
                        person_label=reused_label,
                        session_id=reused_session,
                        first_seen=wall_time,
                        last_seen=wall_time,
                        window_key=window_key,
                        best_confidence=det.confidence,
                        best_crop_path=crop_path,
                        embedding=embedding,
                        last_state="walking",
                        last_bbox=bbox_now,
                        frame_count=1,
                        # Bug 10 fix: always write an entry event for re-acquired tracks.
                        # The person may have been lost across a window boundary — without
                        # a new entry event in the current window, Ask/Search won't see them.
                        track_event_written=False,
                    )
                    # Write entry event for this window (safe even if already exists
                    # — _write_track_event_entry wraps the DB add in try/except)
                    self._write_track_event_entry(track, window_key, write_db)
                    self._write_detected_object(det, track, frame, window_key, write_db)
                else:
                    person_label, session_id = self._identify_or_create(
                        det, embedding, crop_path, wall_time, write_db, read_db)

                    if crop_path and f"tmp{det.track_id}" in crop_path:
                        new_path = crop_path.replace(
                            f"tmp{det.track_id}", person_label)
                        try:
                            os.rename(crop_path, new_path)
                            crop_path = new_path
                            # Fix: sync renamed path back to PersonIdentity so the
                            # crop API endpoint can find the file by its new name.
                            self._update_identity_crop_path(
                                person_label, new_path, write_db)
                        except Exception: pass

                    track = LiveTrack(
                        track_id=det.track_id,
                        person_label=person_label,
                        session_id=session_id,
                        first_seen=wall_time,
                        last_seen=wall_time,
                        window_key=window_key,
                        best_confidence=det.confidence,
                        best_crop_path=crop_path,
                        embedding=embedding,
                        last_state="entering",
                        last_bbox=bbox_now,
                        frame_count=1,
                        objects_nearby=_init_objects,
                        activity_hint=_init_hint,
                    )
                    # Write entry event to TrackEvent (feeds Ask/Search/Timeline)
                    self._write_track_event_entry(track, window_key, write_db)
                    self._write_detected_object(det, track, frame, window_key, write_db)
                    self.logger.info("live_person_entered",
                                     person=person_label,
                                     track_id=det.track_id,
                                     window=window_key,
                                     time=wall_time.strftime("%H:%M:%S"))

                with self._tracks_lock:
                    self._tracks[det.track_id] = track
                # Label is now confirmed in self._tracks — release the reservation
                self._release_reserved(track.person_label)

    def _handle_window_boundary(self, wall_time: datetime, current_window: str,
                                 write_db):
        """
        When the window key changes, split any active cross-boundary tracks:
        write an exit into the old window and an entry into the new window.
        """
        with self._tracks_lock:
            tracks = list(self._tracks.values())

        for track in tracks:
            if track.window_key != current_window:
                # This track started in a previous window — write boundary split
                self.logger.info(
                    "live_window_boundary_split",
                    person=track.person_label,
                    old_window=track.window_key,
                    new_window=current_window,
                )
                # Exit in old window at boundary
                self._write_track_event_exit(track, track.window_key,
                                              wall_time, write_db)
                # Register position briefly so position-match doesn't create
                # a new label for the same person in the new window
                self._register_lost_track(track, wall_time)
                # Entry in new window from boundary
                with self._tracks_lock:
                    track.window_key  = current_window
                    track.first_seen  = wall_time   # reset for new window
                    track.track_event_written = False

                self._write_track_event_entry(track, current_window, write_db)

    def _close_stale_tracks(self, wall_time: datetime, window_key: str, write_db):
        timeout  = self.settings.live_exit_timeout_seconds
        to_close = []
        with self._tracks_lock:
            for tid, track in list(self._tracks.items()):
                if (wall_time - track.last_seen).total_seconds() > timeout:
                    to_close.append((tid, track))

        for tid, track in to_close:
            # Face detection on best keyframe BEFORE exit event — so face
            # fields are already in attributes when exit is written
            self._run_face_detection(track, wall_time, write_db)
            self._write_track_event_exit(track, track.window_key, wall_time, write_db)
            self._close_session(track, wall_time, write_db)
            # Register in position cache so re-acquisition gets same label
            if track.last_bbox != (0, 0, 0, 0):
                self._register_lost_track(track, wall_time)
            with self._tracks_lock:
                self._tracks.pop(tid, None)
            # Clean per-track state dicts to avoid unbounded growth
            self._last_activity_check.pop(tid, None)
            self._last_snapshot_time.pop(tid, None)
            self.logger.info("live_person_exited",
                             person=track.person_label,
                             face_detected=track.face_detected,
                             duration=f"{(wall_time-track.first_seen).total_seconds():.0f}s")

    def _close_all_sessions(self, wall_time: datetime, window_key: str, db):
        with self._tracks_lock:
            tracks = list(self._tracks.values())
        for track in tracks:
            self._run_face_detection(track, wall_time, db)
            self._write_track_event_exit(track, track.window_key, wall_time, db)
            self._close_session(track, wall_time, db)
            if track.last_bbox != (0, 0, 0, 0):
                self._register_lost_track(track, wall_time)

    # ── DB writers ────────────────────────────────────────────────────────────

    def _write_track_event_entry(self, track: LiveTrack, window_key: str, db):
        """
        Write an entry TrackEvent for this live track.
        Uses Unix epoch seconds for first/last seen so duration is always accurate
        regardless of window boundary parsing. Timeline display converts these
        to relative seconds at query time.
        """
        from app.storage.models import TrackEvent
        import time as _time
        try:
            epoch_sec = _time.mktime(track.first_seen.timetuple())

            rag_text = (
                f"{track.person_label} ({track.last_state}) "
                f"entered at {track.first_seen.strftime('%H:%M:%S')} "
                f"in window {window_key}."
            )
            ev = TrackEvent(
                video_filename=window_key,
                camera_id=self.settings.camera_id,
                track_id=track.track_id,
                object_class="person",
                event_type="entry",
                first_seen_second=epoch_sec,
                last_seen_second=epoch_sec,
                duration_seconds=0.0,
                best_frame_second=epoch_sec,
                best_crop_path=track.best_crop_path,
                best_confidence=track.best_confidence,
                rag_text=rag_text,
                attributes={
                    "person_label":    track.person_label,
                    "live":            True,
                    "entry_wall_time": track.first_seen.isoformat(),
                    "objects_nearby":  [],
                    "activity_hint":   "present",
                },
            )
            db.add(ev)
            db.flush()   # get ev.id without full commit

            # Write initial embedding immediately so Ask/Search works while
            # the window is still LIVE (not just after it closes + reembed).
            # _run_reembed() will overwrite this with the enriched version later.
            try:
                from app.storage.models import TrackEventEmbedding
                from app.rag.embedder import OllamaEmbedder
                vec = OllamaEmbedder().embed(ev.rag_text)
                if vec is not None:
                    db.add(TrackEventEmbedding(
                        track_event_id=ev.id,
                        embedding=vec,
                        model_name=self.settings.embed_model
                            if hasattr(self.settings, "embed_model") else "nomic-embed-text",
                    ))
            except Exception:
                pass  # embedding failure must never block entry write

            db.commit()
            with self._tracks_lock:
                track.track_event_written = True
        except Exception as e:
            self.logger.debug("track_event_entry_failed", error=str(e))
            try: db.rollback()
            except Exception: pass

    def _write_track_event_exit(self, track: LiveTrack, window_key: str,
                                 exit_time: datetime, db):
        """
        Finalise the TrackEvent with accurate wall-clock duration.
        duration_seconds = (exit_time - first_seen).total_seconds()
        This is what TemporalAnalyzer reads to classify behaviour.
        """
        from app.storage.models import TrackEvent
        from sqlalchemy import and_
        import time as _time
        try:
            dur      = (exit_time - track.first_seen).total_seconds()
            last_sec = _time.mktime(exit_time.timetuple())

            ev = db.query(TrackEvent).filter(
                and_(
                    TrackEvent.video_filename == window_key,
                    TrackEvent.track_id       == track.track_id,
                    TrackEvent.event_type     == "entry",
                )).first()

            if ev:
                ev.last_seen_second  = last_sec
                ev.duration_seconds  = max(0.0, dur)
                ev.best_crop_path    = track.best_crop_path or ev.best_crop_path
                ev.best_confidence   = track.best_confidence or ev.best_confidence
                ev.rag_text = (
                    f"{track.person_label} ({track.last_state}) "
                    f"was present from {track.first_seen.strftime('%H:%M:%S')} "
                    f"to {exit_time.strftime('%H:%M:%S')} "
                    f"({int(dur)}s) in window {window_key}."
                )
                if ev.attributes:
                    ev.attributes = dict(
                        ev.attributes,
                        exit_wall_time=exit_time.isoformat(),
                        last_state=track.last_state,
                        duration_seconds=round(dur, 1),
                        objects_nearby=track.objects_nearby if hasattr(track, "objects_nearby") else [],
                        activity_hint=track.activity_hint if hasattr(track, "activity_hint") else "present",
                        face_detected=getattr(track, "face_detected", False),
                        face_crop_path=getattr(track, "face_crop_path", None),
                        face_confidence=getattr(track, "face_confidence", 0.0),
                    )
                from sqlalchemy.orm.attributes import flag_modified as _fm
                _fm(ev, "attributes")
                db.commit()
        except Exception as e:
            self.logger.debug("track_event_exit_failed", error=str(e))
            try: db.rollback()
            except Exception: pass

    def _update_track_event(self, track: LiveTrack, db):
        """Periodic rolling update: keeps duration accurate as person stays in frame."""
        from app.storage.models import TrackEvent
        from sqlalchemy import and_
        import time as _time
        try:
            now = datetime.utcnow()
            dur = (now - track.first_seen).total_seconds()

            ev = db.query(TrackEvent).filter(
                and_(
                    TrackEvent.video_filename == track.window_key,
                    TrackEvent.track_id       == track.track_id,
                    TrackEvent.event_type     == "entry",
                )).first()
            if ev:
                ev.last_seen_second = _time.mktime(now.timetuple())
                ev.duration_seconds = max(0.0, dur)
                if ev.attributes:
                    ev.attributes = dict(
                        ev.attributes,
                        duration_seconds=round(dur, 1),
                        last_state=track.last_state,
                        objects_nearby=track.objects_nearby if hasattr(track, "objects_nearby") else [],
                        activity_hint=track.activity_hint if hasattr(track, "activity_hint") else "present",
                    )
                from sqlalchemy.orm.attributes import flag_modified as _fm
                _fm(ev, "attributes")
                db.commit()
        except Exception as e:
            self.logger.debug("track_event_update_failed", error=str(e))
            try: db.rollback()
            except Exception: pass

    def _write_detected_object(self, det, track: LiveTrack, frame,
                                window_key: str, db):
        """Write DetectedObject row so Detections tab shows live data."""
        from app.storage.models import DetectedObject
        try:
            h, w = frame.shape[:2]
            do = DetectedObject(
                video_filename=window_key,
                camera_id=self.settings.camera_id,
                track_id=det.track_id,
                object_class=det.object_class,
                confidence=det.confidence,
                frame_second_offset=0.0,
                bbox_x1=det.x1 / w, bbox_y1=det.y1 / h,
                bbox_x2=det.x2 / w, bbox_y2=det.y2 / h,
                crop_path=track.best_crop_path,
                frame_quadrant="center",
                rag_text=f"Live detection: {track.person_label} in {window_key}",
            )
            db.add(do)
            db.commit()
        except Exception as e:
            self.logger.debug("detected_object_write_failed", error=str(e))
            try: db.rollback()
            except Exception: pass

    # ── Person identity (ReID + sessions) ─────────────────────────────────────


    def _run_face_detection(self, track: "LiveTrack", wall_time: datetime, db):
        """
        Run face detection on the track's best keyframe and persist results.
        Called once at track close — uses the highest-quality crop accumulated
        over the full track duration, not the first blurry entry frame.
        """
        if not track.best_crop_path:
            return
        try:
            from app.storage.models import TrackEvent
            from sqlalchemy import and_
            from sqlalchemy.orm.attributes import flag_modified

            faces_dir = self.settings.live_crops_path.rstrip("/") + "/faces"
            result = self.face_detector.detect(
                crop_path=track.best_crop_path,
                save_dir=faces_dir,
                label=track.person_label,
                wall_time=wall_time,
            )

            # Store on in-memory track so exit writer picks them up
            track.face_detected   = result.detected
            track.face_crop_path  = result.face_crop_path
            track.face_confidence = result.confidence

            # Also update the TrackEvent.attributes immediately
            ev = db.query(TrackEvent).filter(
                and_(
                    TrackEvent.video_filename == track.window_key,
                    TrackEvent.track_id       == track.track_id,
                    TrackEvent.event_type     == "entry",
                )
            ).first()
            if ev:
                attrs = dict(ev.attributes or {})
                attrs["face_detected"]   = result.detected
                attrs["face_crop_path"]  = result.face_crop_path
                attrs["face_confidence"] = result.confidence
                attrs["face_method"]     = result.method
                ev.attributes = attrs
                flag_modified(ev, "attributes")
                db.commit()

            self.logger.info(
                "face_detection_done",
                person=track.person_label,
                detected=result.detected,
                method=result.method,
                confidence=result.confidence,
            )
        except Exception as e:
            self.logger.debug("face_detection_failed",
                              person=track.person_label, error=str(e))
            try: db.rollback()
            except Exception: pass

    def _match_by_position(self, bbox: tuple, wall_time: datetime,
                            merge_gap_seconds: int = 30) -> tuple:
        """
        POSITION-FIRST re-identification for static scenes (offices, desks).

        If a new ByteTrack ID appears at position (cx,cy) and there was a
        recently-lost track within spatial_radius pixels → same physical person.
        Works even when appearance embeddings are unreliable (dark, similar clothing).

        spatial_radius: scales with person height.
          box_h * 0.4 covers normal seated fidgeting + slight camera shake.
          At typical desk-cam resolution, box_h ≈ 200px → radius ≈ 80px.

        Returns (person_label, session_id) if position match found, else (None, None).
        """
        cx = (bbox[0] + bbox[2]) / 2
        cy = (bbox[1] + bbox[3]) / 2
        bh = bbox[3] - bbox[1]
        bw = bbox[2] - bbox[0]
        spatial_radius = max(bh, bw) * 0.45   # 45% of person size

        now = wall_time
        best_dist  = float("inf")
        best_label = None
        best_sid   = None

        with self._lost_tracks_lock:
            expired = [
                lbl for lbl, info in self._lost_tracks.items()
                if (now - info["last_seen"]).total_seconds() > merge_gap_seconds
            ]
            for lbl in expired:
                del self._lost_tracks[lbl]

            for label, info in self._lost_tracks.items():
                # Skip if this label is already active
                with self._tracks_lock:
                    active_labels = {t.person_label for t in self._tracks.values()}
                if label in active_labels or label in self._reserved_labels:
                    continue

                age = (now - info["last_seen"]).total_seconds()
                if age > merge_gap_seconds:
                    continue

                dist = ((cx - info["cx"])**2 + (cy - info["cy"])**2) ** 0.5
                if dist < spatial_radius and dist < best_dist:
                    best_dist  = dist
                    best_label = label
                    best_sid   = info["session_id"]

        if best_label:
            self.logger.info(
                "live_position_match",
                person=best_label,
                dist_px=f"{best_dist:.0f}",
                radius_px=f"{spatial_radius:.0f}",
            )
            # Remove from lost — it's active again
            with self._lost_tracks_lock:
                self._lost_tracks.pop(best_label, None)

        return best_label, best_sid

    def _register_lost_track(self, track: "LiveTrack", wall_time: datetime):
        """Save a just-exited track to the position cache for future re-identification."""
        cx = (track.last_bbox[0] + track.last_bbox[2]) / 2
        cy = (track.last_bbox[1] + track.last_bbox[3]) / 2
        bh = track.last_bbox[3] - track.last_bbox[1]
        with self._lost_tracks_lock:
            self._lost_tracks[track.person_label] = {
                "cx":         cx,
                "cy":         cy,
                "box_h":      bh,
                "last_seen":  wall_time,
                "session_id": track.session_id,
                "embedding":  track.embedding,
            }

    def _match_active_track(self, embedding, new_track_id, new_bbox,
                             threshold=0.85):
        now     = datetime.utcnow()
        timeout = self.settings.live_exit_timeout_seconds * 3
        with self._tracks_lock:
            candidates = [
                t for t in self._tracks.values()
                if t.track_id != new_track_id
                and t.embedding is not None
                and (now - t.last_seen).total_seconds() < timeout
            ]
        best_score, best_track = 0.0, None
        for t in candidates:
            try:
                score = float(np.dot(embedding, t.embedding) /
                              (np.linalg.norm(embedding) *
                               np.linalg.norm(t.embedding) + 1e-8))
                if score <= best_score:
                    continue
                # Spatial conflict check:
                # If candidate is still fresh (within exit timeout) and their
                # bounding box centres are far apart → different people.
                # Threshold: 0.5 * person height — tight enough for dense scenes.
                if new_bbox and t.last_bbox != (0,0,0,0):
                    age = (now - t.last_seen).total_seconds()
                    if age < self.settings.live_exit_timeout_seconds:
                        cx1 = (new_bbox[0]+new_bbox[2])/2
                        cy1 = (new_bbox[1]+new_bbox[3])/2
                        cx2 = (t.last_bbox[0]+t.last_bbox[2])/2
                        cy2 = (t.last_bbox[1]+t.last_bbox[3])/2
                        dist  = ((cx1-cx2)**2+(cy1-cy2)**2)**0.5
                        box_h = max(
                            t.last_bbox[3] - t.last_bbox[1],
                            new_bbox[3]    - new_bbox[1],
                        )
                        # 0.5 person-height separation → definitively different people
                        if box_h > 0 and dist > box_h * 0.5:
                            continue
                best_score = score
                best_track = t
            except Exception:
                continue
        if best_track and best_score >= threshold:
            return best_track.person_label, best_track.session_id
        return None, None

    def _identify_or_create(self, det, embedding, crop_path, wall_time,
                             write_db, read_db):
        from app.storage.live_models import PersonIdentity, PersonSession
        import uuid as _uuid

        threshold        = self.settings.live_reid_threshold
        best_match_label = None
        best_score       = 0.0
        now_dt           = datetime.utcnow()

        if embedding is not None:
            try:
                for ident in read_db.query(PersonIdentity).all():
                    if not ident.embedding:
                        continue
                    stored = np.array(ident.embedding, dtype=np.float32)
                    n1 = np.linalg.norm(embedding)
                    n2 = np.linalg.norm(stored)
                    if n1 < 1e-6 or n2 < 1e-6:
                        continue
                    score = float(np.dot(embedding, stored) / (n1 * n2))
                    if score > best_score:
                        best_score       = score
                        best_match_label = ident.person_label
            except Exception as e:
                self.logger.warning("reid_query_failed", error=str(e))

        # Spatial conflict: block match if same label active at different position
        if best_match_label and best_score >= threshold:
            with self._tracks_lock:
                conflict = [
                    t for t in self._tracks.values()
                    if t.person_label == best_match_label
                    and t.last_bbox != (0,0,0,0)
                    and (now_dt - t.last_seen).total_seconds()
                        < self.settings.live_exit_timeout_seconds
                ]
            for ct in conflict:
                cx1 = (det.x1+det.x2)/2; cy1 = (det.y1+det.y2)/2
                cx2 = (ct.last_bbox[0]+ct.last_bbox[2])/2
                cy2 = (ct.last_bbox[1]+ct.last_bbox[3])/2
                dist  = ((cx1-cx2)**2+(cy1-cy2)**2)**0.5
                # Use max of both boxes for dense scenes
                box_h = max(
                    ct.last_bbox[3] - ct.last_bbox[1],
                    det.y2 - det.y1,
                )
                if box_h > 0 and dist > box_h * 0.5:
                    best_match_label = None
                    break

        self.logger.info("reid_result", match=best_match_label,
                         score=f"{best_score:.3f}", threshold=threshold)

        if best_match_label and best_score >= threshold:
            person_label = best_match_label
            try:
                ident = write_db.query(PersonIdentity).filter(
                    PersonIdentity.person_label == person_label).first()
                if ident:
                    ident.total_visits += 1
                    ident.last_seen_at  = wall_time
                    if crop_path and det.confidence > 0.7:
                        ident.best_crop_path = crop_path
                    if embedding is not None and ident.embedding:
                        stored  = np.array(ident.embedding, dtype=np.float32)
                        blended = 0.8 * stored + 0.2 * embedding
                        n       = np.linalg.norm(blended)
                        ident.embedding = (blended/n if n>0 else blended).tolist()
                    write_db.commit()
            except Exception as e:
                self.logger.warning("identity_update_failed", error=str(e))
                try: write_db.rollback()
                except Exception: pass
        else:
            person_label = self._next_label()
            try:
                ident = PersonIdentity(
                    person_label=person_label,
                    embedding=embedding.tolist() if embedding is not None else None,
                    best_crop_path=crop_path,
                    first_seen_at=wall_time,
                    last_seen_at=wall_time,
                    total_visits=1,
                )
                write_db.add(ident)
                write_db.commit()
            except Exception as e:
                self.logger.error("identity_create_failed",
                                  label=person_label, error=str(e))
                try: write_db.rollback()
                except Exception: pass
                # Release reservation so this label isn't permanently blocked
                self._release_reserved(person_label)

        session_id = str(_uuid.uuid4())
        try:
            session = PersonSession(
                id=_uuid.UUID(session_id),
                person_label=person_label,
                track_id=det.track_id,
                entry_time=wall_time,
                last_state="entering",
                best_crop_path=crop_path,
                best_confidence=det.confidence,
                is_active=True,
            )
            write_db.add(session)
            write_db.commit()
        except Exception as e:
            self.logger.error("session_create_failed", error=str(e))
            try: write_db.rollback()
            except Exception: pass

        return person_label, session_id

    def _persist_session(self, track: LiveTrack, db):
        from app.storage.live_models import PersonSession
        import uuid as _uuid
        try:
            session = db.query(PersonSession).filter(
                PersonSession.id == _uuid.UUID(track.session_id)).first()
            if session:
                session.exit_time        = track.last_seen
                session.duration_seconds = (track.last_seen - track.first_seen).total_seconds()
                session.last_state       = track.last_state
                session.best_crop_path   = track.best_crop_path
                db.commit()
        except Exception as e:
            self.logger.debug("session_persist_failed", error=str(e))
            try: db.rollback()
            except Exception: pass

    def _close_session(self, track: LiveTrack, exit_time: datetime, db):
        from app.storage.live_models import PersonSession
        import uuid as _uuid
        try:
            session = db.query(PersonSession).filter(
                PersonSession.id == _uuid.UUID(track.session_id)).first()
            if session:
                session.exit_time        = exit_time
                session.duration_seconds = (exit_time - track.first_seen).total_seconds()
                session.last_state       = track.last_state
                session.best_crop_path   = track.best_crop_path
                session.is_active        = False
                db.commit()
        except Exception as e:
            self.logger.debug("session_close_failed", error=str(e))
            try: db.rollback()
            except Exception: pass

    def _update_identity_embedding(self, person_label: str, crop_path: str, db):
        from app.storage.live_models import PersonIdentity
        try:
            new_emb = self.reid.embed_crop(crop_path)
            if new_emb is None: return
            ident = db.query(PersonIdentity).filter(
                PersonIdentity.person_label == person_label).first()
            if ident and ident.embedding:
                stored  = np.array(ident.embedding, dtype=np.float32)
                blended = 0.7 * stored + 0.3 * new_emb
                n       = np.linalg.norm(blended)
                ident.embedding      = (blended/n if n>0 else blended).tolist()
                # Fix: keep best_crop_path in sync with the highest-confidence
                # crop — previously only the embedding was updated here.
                ident.best_crop_path = crop_path
                db.commit()
        except Exception:
            try: db.rollback()
            except Exception: pass

    def _update_identity_crop_path(self, person_label: str, crop_path: str, db):
        """Persist best_crop_path to PersonIdentity after a rename or better crop."""
        from app.storage.live_models import PersonIdentity
        try:
            ident = db.query(PersonIdentity).filter(
                PersonIdentity.person_label == person_label).first()
            if ident:
                ident.best_crop_path = crop_path
                db.commit()
        except Exception:
            try: db.rollback()
            except Exception: pass

    # ── Frame annotation (with CLAHE brightness) ──────────────────────────────

    def _enhance_brightness(self, frame: np.ndarray) -> np.ndarray:
        try:
            lab  = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            cl    = clahe.apply(l)
            return cv2.cvtColor(cv2.merge((cl, a, b)), cv2.COLOR_LAB2BGR)
        except Exception:
            return frame.copy()

    def _annotate_frame(self, frame: np.ndarray) -> np.ndarray:
        out = self._enhance_brightness(frame)
        h, w = out.shape[:2]

        # Timestamp
        ts = datetime.utcnow().strftime("%d-%m-%Y  %H:%M:%S  UTC")
        cv2.putText(out, ts, (w - 330, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (200,200,200), 1, cv2.LINE_AA)
        # LIVE badge
        cv2.circle(out, (20, 20), 8, (0, 0, 220), -1)
        cv2.putText(out, "LIVE", (34, 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220,220,220), 1, cv2.LINE_AA)

        now     = datetime.utcnow()
        timeout = self.settings.live_exit_timeout_seconds
        with self._tracks_lock:
            tracks = list(self._tracks.values())

        for track in tracks:
            age = (now - track.last_seen).total_seconds()
            if age > timeout or track.last_bbox == (0,0,0,0):
                continue
            x1, y1, x2, y2 = [int(v) for v in track.last_bbox]
            color = _color_for(track.person_label)
            cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
            # Show activity hint if available, otherwise state
            activity = track.activity_hint if track.activity_hint and track.activity_hint != "present" else track.last_state
            label = f"{track.person_label}  {activity.upper()}"
            (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
            cv2.rectangle(out, (x1, y1-lh-10), (x1+lw+8, y1), color, -1)
            cv2.putText(out, label, (x1+4, y1-4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,0,0), 1, cv2.LINE_AA)
            dur = (track.last_seen - track.first_seen).total_seconds()
            dur_str = f"{int(dur)}s" if dur < 60 else f"{int(dur//60)}m{int(dur%60)}s"
            cv2.putText(out, dur_str, (x1+4, y2-6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
        return out

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _next_label(self) -> str:
        """
        Returns next available label, skipping both:
          - Labels currently active in self._tracks (confirmed tracks)
          - Labels in self._reserved_labels (assigned this batch, not yet in tracks)
        The caller MUST call _release_reserved(label) if the label ends up
        unused (e.g. on DB failure), or it stays reserved until next restart.
        """
        with self._label_lock:
            while True:
                label = f"P{self._next_label_num}"
                self._next_label_num += 1
                with self._tracks_lock:
                    active = {t.person_label for t in self._tracks.values()}
                if label not in active and label not in self._reserved_labels:
                    self._reserved_labels.add(label)
                    return label

    def _release_reserved(self, label: str):
        """Remove a label from the reserved set (call when track confirmed in self._tracks)."""
        with self._label_lock:
            self._reserved_labels.discard(label)

    def _init_label_counter(self, db):
        from app.storage.live_models import PersonIdentity
        try:
            nums = []
            for ident in db.query(PersonIdentity).all():
                try: nums.append(int(ident.person_label.lstrip("P")))
                except ValueError: pass
            if nums:
                with self._label_lock:
                    self._next_label_num = max(nums) + 1
        except Exception as e:
            self.logger.warning("label_init_failed", error=str(e))

    def _save_crop(self, crop: np.ndarray, label: str,
                    wall_time: datetime) -> Optional[str]:
        try:
            d = self.settings.live_crops_path
            os.makedirs(d, exist_ok=True)
            ts   = wall_time.strftime("%Y%m%d_%H%M%S_%f")
            path = os.path.join(d, f"{label}_{ts}.jpg")
            cv2.imwrite(path, crop, [cv2.IMWRITE_JPEG_QUALITY, 85])
            return path
        except Exception:
            return None