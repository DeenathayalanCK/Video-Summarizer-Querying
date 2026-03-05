"""
live_stream_processor.py — Continuous RTSP stream processing.

Runs as a background daemon thread. Reads RTSP frames, runs YOLO+ByteTrack,
manages PersonSessions with wall-clock timestamps, produces MJPEG stream.

KEY DESIGN DECISIONS:

  Person identity is tracked at TWO levels:
    - ByteTrack ID: assigned per-frame by ByteTrack. Resets whenever tracker
      loses the person. Same physical person can get 5 different ByteTrack IDs.
    - Person label (P1, P2...): persistent human-readable ID. Assigned once
      per physical person, reused across ByteTrack ID changes.

  ReID strategy (appearance + session merge window):
    1. If new ByteTrack ID appears and its embedding matches an ACTIVE in-memory
       track's embedding (same session, just tracker glitch) → reuse label + session.
    2. If embedding matches a KNOWN identity in DB above threshold → returning person.
    3. Otherwise → new person, assign next available label.

  DB safety:
    - Fresh DB session per-frame for writes (avoids stale session state).
    - Long-lived session for reads only.
    - All writes wrapped in try/except with rollback.
"""

import os
import cv2
import threading
import time
import numpy as np
from datetime import datetime, timedelta
from typing import Optional, Dict
from dataclasses import dataclass, field

from app.core.config import get_settings
from app.core.logging import get_logger
from app.detection.detector import ObjectDetector, Detection
from app.detection.appearance_reid import AppearanceReID
from app.detection.crop_utils import extract_crop


# ── Per-track state ───────────────────────────────────────────────────────────

@dataclass
class LiveTrack:
    track_id:        int
    person_label:    str
    session_id:      str
    first_seen:      datetime
    last_seen:       datetime
    best_confidence: float = 0.0
    best_crop_path:  Optional[str] = None
    embedding:       Optional[np.ndarray] = None
    last_state:      str = "unknown"
    last_bbox:       tuple = (0, 0, 0, 0)
    frame_count:     int = 0


# ── Colors / annotations ──────────────────────────────────────────────────────

_COLORS = [
    (0, 165, 255), (0, 255, 0), (255, 0, 255), (0, 255, 255),
    (255, 128, 0), (128, 0, 255), (0, 128, 255), (255, 0, 128),
    (255, 200, 0), (0, 200, 255),
]

def _color_for(label: str) -> tuple:
    idx = sum(ord(c) for c in label)
    return _COLORS[idx % len(_COLORS)]


def _classify_state(bbox_now, bbox_prev, frame_count: int) -> str:
    if bbox_prev is None or frame_count < 3:
        return "entering"
    cx_n = (bbox_now[0]  + bbox_now[2])  / 2
    cy_n = (bbox_now[1]  + bbox_now[3])  / 2
    cx_p = (bbox_prev[0] + bbox_prev[2]) / 2
    cy_p = (bbox_prev[1] + bbox_prev[3]) / 2
    dist = ((cx_n - cx_p)**2 + (cy_n - cy_p)**2) ** 0.5
    box_diag = ((bbox_now[2]-bbox_now[0])**2 + (bbox_now[3]-bbox_now[1])**2) ** 0.5
    if box_diag == 0:
        return "stationary"
    ratio = dist / box_diag
    if ratio > 0.30:  return "running"
    if ratio > 0.07:  return "walking"
    return "stationary"


# ── Frame buffer ──────────────────────────────────────────────────────────────

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
        self.settings = get_settings()
        self.logger   = get_logger()
        self.detector = ObjectDetector()
        self.reid     = AppearanceReID()

        self._thread:  Optional[threading.Thread] = None
        self._stop_evt = threading.Event()
        self._running  = False

        self.frame_buffer = FrameBuffer()

        # {bytetrack_id → LiveTrack}
        self._tracks:      Dict[int, LiveTrack] = {}
        self._tracks_lock  = threading.Lock()

        # Persists known labels between restarts
        # {person_label → last_seen datetime}
        self._known_labels: Dict[str, datetime] = {}

        self._next_label_num = 1
        self._label_lock     = threading.Lock()

    # ── Control ───────────────────────────────────────────────────────────────

    def start(self) -> bool:
        if self._running:
            return False
        if not self.settings.rtsp_url:
            self.logger.error("live_start_failed", reason="RTSP_URL not set")
            return False
        self._stop_evt.clear()
        self._thread = threading.Thread(
            target=self._run_loop, daemon=True, name="live_stream")
        self._thread.start()
        self._running = True
        self.logger.info("live_stream_started", url=self.settings.rtsp_url)
        return True

    def stop(self):
        self._stop_evt.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=10)
        self._running = False
        self.logger.info("live_stream_stopped")

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
                    })
            return sorted(result, key=lambda x: x["person_label"])

    # ── Main loop ─────────────────────────────────────────────────────────────

    def _run_loop(self):
        sample_interval = 1.0 / max(1, self.settings.live_sample_fps)
        quality         = self.settings.live_stream_jpeg_quality

        from app.storage.database import SessionLocal
        read_db = SessionLocal()   # long-lived session for reads
        try:
            self._init_label_counter(read_db)
        except Exception as e:
            self.logger.warning("live_label_init_failed", error=str(e))

        while not self._stop_evt.is_set():
            cap = None
            try:
                self.logger.info("live_connecting", url=self.settings.rtsp_url)
                cap = cv2.VideoCapture(self.settings.rtsp_url)
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)
                if not cap.isOpened():
                    raise RuntimeError("Cannot open RTSP stream")
                self.logger.info("live_connected")

                last_sample = 0.0
                while not self._stop_evt.is_set():
                    ret, frame = cap.read()
                    if not ret:
                        self.logger.warning("live_frame_read_failed")
                        break

                    now = time.time()

                    # Always push annotated frame even between detection samples
                    annotated = self._annotate_frame(frame)
                    ok, buf = cv2.imencode(
                        ".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, quality])
                    if ok:
                        self.frame_buffer.put(bytes(buf))

                    if now - last_sample < sample_interval:
                        continue
                    last_sample = now
                    wall_time   = datetime.utcnow()

                    # Detection
                    try:
                        dets = self.detector.detect_with_tracking(frame, now)
                    except Exception as e:
                        self.logger.warning("live_detect_error", error=str(e))
                        continue

                    if dets and dets.has_objects:
                        # Fresh write session per detection batch
                        write_db = SessionLocal()
                        try:
                            self._update_tracks(frame, dets, wall_time, write_db, read_db)
                            self._close_stale_tracks(wall_time, write_db)
                        except Exception as e:
                            self.logger.error("live_track_update_error", error=str(e))
                            try:
                                write_db.rollback()
                            except Exception:
                                pass
                        finally:
                            write_db.close()
                    else:
                        write_db = SessionLocal()
                        try:
                            self._close_stale_tracks(wall_time, write_db)
                        finally:
                            write_db.close()

            except Exception as e:
                self.logger.error("live_loop_error", error=str(e))
            finally:
                if cap:
                    cap.release()

            if not self._stop_evt.is_set():
                self.logger.info("live_reconnecting_in_5s")
                self._stop_evt.wait(timeout=5)

        # Shutdown: close all sessions
        shutdown_db = SessionLocal()
        try:
            self._close_all_sessions(datetime.utcnow(), shutdown_db)
        except Exception:
            pass
        finally:
            shutdown_db.close()
            read_db.close()
            self._running = False

    # ── Track management ──────────────────────────────────────────────────────

    def _update_tracks(self, frame, dets, wall_time: datetime, write_db, read_db):
        """Process detections from one frame."""

        for det in dets.detections:
            if det.object_class != "person" or det.track_id is None:
                continue

            bbox_now = (det.x1, det.y1, det.x2, det.y2)

            with self._tracks_lock:
                existing = self._tracks.get(det.track_id)

            if existing:
                # ── Known ByteTrack ID — just update ──────────────────────────
                prev_bbox = existing.last_bbox
                state     = _classify_state(bbox_now, prev_bbox, existing.frame_count)
                with self._tracks_lock:
                    existing.last_seen   = wall_time
                    existing.last_state  = state
                    existing.last_bbox   = bbox_now
                    existing.frame_count += 1

                if det.confidence > (existing.best_confidence or 0) + 0.08:
                    crop = extract_crop(frame, det)
                    if crop is not None and crop.size > 0:
                        path = self._save_crop(crop, existing.person_label, wall_time)
                        if path:
                            with self._tracks_lock:
                                existing.best_crop_path  = path
                                existing.best_confidence = det.confidence
                            self._update_embedding_db(
                                existing.person_label, path, write_db)

                # Periodic DB update (every 10 frames)
                if existing.frame_count % 10 == 0:
                    self._persist_session(existing, write_db)

            else:
                # ── New ByteTrack ID — could be new person or re-tracked ───────

                # Extract crop and embedding first
                crop = extract_crop(frame, det)
                crop_path = None
                embedding = None
                if crop is not None and crop.size > 0:
                    # Temp save to embed, rename after label assigned
                    crop_path = self._save_crop(crop, f"tmp{det.track_id}", wall_time)
                    if crop_path:
                        embedding = self.reid.embed_crop(crop_path)

                # Step 1: check if this matches any CURRENTLY ACTIVE in-memory track
                # (same physical person whose ByteTrack ID just changed)
                reused_label    = None
                reused_session  = None

                if embedding is not None:
                    reused_label, reused_session = self._match_active_track(
                        embedding, det.track_id, new_bbox=bbox_now)

                if reused_label:
                    # Tracker glitch — same person, new ByteTrack ID
                    # Update the old track entry with the new track_id
                    self.logger.info(
                        "live_track_relinked",
                        person=reused_label,
                        old_id="?", new_id=det.track_id,
                    )
                    track = LiveTrack(
                        track_id=det.track_id,
                        person_label=reused_label,
                        session_id=reused_session,
                        first_seen=wall_time,
                        last_seen=wall_time,
                        best_confidence=det.confidence,
                        best_crop_path=crop_path,
                        embedding=embedding,
                        last_state="walking",
                        last_bbox=bbox_now,
                        frame_count=1,
                    )
                else:
                    # Step 2: match against DB (returning person)
                    person_label, session_id = self._identify_or_create(
                        det, embedding, crop_path, wall_time, write_db, read_db)

                    # Rename temp crop
                    if crop_path and f"tmp{det.track_id}" in crop_path and person_label not in crop_path:
                        new_path = crop_path.replace(f"tmp{det.track_id}", person_label)
                        try:
                            os.rename(crop_path, new_path)
                            crop_path = new_path
                        except Exception:
                            pass

                    track = LiveTrack(
                        track_id=det.track_id,
                        person_label=person_label,
                        session_id=session_id,
                        first_seen=wall_time,
                        last_seen=wall_time,
                        best_confidence=det.confidence,
                        best_crop_path=crop_path,
                        embedding=embedding,
                        last_state="entering",
                        last_bbox=bbox_now,
                        frame_count=1,
                    )
                    self.logger.info(
                        "live_person_entered",
                        person=person_label,
                        track_id=det.track_id,
                        time=wall_time.strftime("%H:%M:%S"),
                    )

                with self._tracks_lock:
                    self._tracks[det.track_id] = track
                with self._label_lock:
                    self._known_labels[track.person_label] = wall_time

    def _match_active_track(
        self, embedding: np.ndarray, new_track_id: int,
        new_bbox: tuple = None,
        similarity_threshold: float = 0.85
    ) -> tuple:
        """
        Check if this embedding matches a recently-lost active track
        (same physical person whose ByteTrack ID just changed).

        IMPORTANT: Only matches if:
          1. Embedding similarity >= threshold (appearance match)
          2. The candidate track is NOT currently active with a DIFFERENT bbox
             at a far spatial distance (two people in same frame can't be same)
          3. The candidate track was seen recently (within 3x exit timeout)

        This prevents: Person A at desk + Person B at door both getting P1
        just because they look similar in low-light footage.
        """
        now     = datetime.utcnow()
        timeout = self.settings.live_exit_timeout_seconds * 3

        with self._tracks_lock:
            candidates = [
                t for t in self._tracks.values()
                if t.track_id != new_track_id
                and t.embedding is not None
                and (now - t.last_seen).total_seconds() < timeout
            ]

        best_score = 0.0
        best_track = None
        for t in candidates:
            try:
                score = float(np.dot(embedding, t.embedding) /
                              (np.linalg.norm(embedding) * np.linalg.norm(t.embedding) + 1e-8))
                if score <= best_score:
                    continue

                # Spatial sanity: if candidate is STILL ACTIVE (seen < exit_timeout ago)
                # and the bboxes are far apart — these are TWO DIFFERENT people
                if new_bbox and t.last_bbox != (0,0,0,0):
                    age = (now - t.last_seen).total_seconds()
                    if age < self.settings.live_exit_timeout_seconds:
                        # Both active simultaneously — check spatial distance
                        cx1 = (new_bbox[0] + new_bbox[2]) / 2
                        cy1 = (new_bbox[1] + new_bbox[3]) / 2
                        cx2 = (t.last_bbox[0] + t.last_bbox[2]) / 2
                        cy2 = (t.last_bbox[1] + t.last_bbox[3]) / 2
                        dist = ((cx1-cx2)**2 + (cy1-cy2)**2)**0.5
                        box_h = t.last_bbox[3] - t.last_bbox[1]
                        # If centres are more than 1 person-height apart → different people
                        if box_h > 0 and dist > box_h * 0.8:
                            continue   # spatial conflict — skip this candidate

                best_score = score
                best_track = t
            except Exception:
                continue

        if best_track and best_score >= similarity_threshold:
            self.logger.info(
                "live_active_track_match",
                person=best_track.person_label,
                score=f"{best_score:.3f}",
            )
            return best_track.person_label, best_track.session_id

        return None, None

    def _identify_or_create(
        self, det, embedding, crop_path, wall_time: datetime,
        write_db, read_db
    ) -> tuple:
        """
        Match against known PersonIdentity in DB, or create new identity.
        Returns (person_label, session_id).
        All DB ops wrapped in try/except with rollback on failure.
        """
        from app.storage.live_models import PersonIdentity, PersonSession
        import uuid as _uuid

        threshold = self.settings.live_reid_threshold
        # In poor lighting all people look similar — also check that this
        # label is NOT currently active in a different spatial position
        now_dt = datetime.utcnow()
        best_match_label  = None
        best_score        = 0.0

        # ── DB ReID: compare against known identities ─────────────────────────
        if embedding is not None:
            try:
                identities = read_db.query(PersonIdentity).all()
                for ident in identities:
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
                self.logger.warning("live_reid_query_failed", error=str(e))

        # Spatial conflict check: if best_match is currently ACTIVE in self._tracks
        # at a different position from det's bbox → cannot be same person
        if best_match_label and best_score >= threshold:
            with self._tracks_lock:
                conflict = [
                    t for t in self._tracks.values()
                    if t.person_label == best_match_label
                    and t.last_bbox != (0,0,0,0)
                    and (now_dt - t.last_seen).total_seconds() < self.settings.live_exit_timeout_seconds
                ]
            for ct in conflict:
                cx1 = (det.x1 + det.x2) / 2
                cy1 = (det.y1 + det.y2) / 2
                cx2 = (ct.last_bbox[0] + ct.last_bbox[2]) / 2
                cy2 = (ct.last_bbox[1] + ct.last_bbox[3]) / 2
                dist = ((cx1-cx2)**2 + (cy1-cy2)**2)**0.5
                box_h = ct.last_bbox[3] - ct.last_bbox[1]
                if box_h > 0 and dist > box_h * 0.6:
                    self.logger.info(
                        "live_reid_spatial_conflict",
                        blocked_label=best_match_label,
                        score=f"{best_score:.3f}",
                        dist=f"{dist:.0f}px",
                    )
                    best_match_label = None  # force new label
                    break

        self.logger.info(
            "live_reid_result",
            best_match=best_match_label,
            score=f"{best_score:.3f}",
            threshold=threshold,
            has_embedding=embedding is not None,
        )

        if best_match_label and best_score >= threshold:
            # ── Returning known person ────────────────────────────────────────
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
                        ident.embedding = (blended / n if n > 0 else blended).tolist()
                    write_db.commit()
            except Exception as e:
                self.logger.warning("live_identity_update_failed", error=str(e))
                try: write_db.rollback()
                except Exception: pass

        else:
            # ── New person — assign next label ────────────────────────────────
            person_label = self._next_label()
            emb_list     = embedding.tolist() if embedding is not None else None
            try:
                ident = PersonIdentity(
                    person_label=person_label,
                    embedding=emb_list,
                    best_crop_path=crop_path,
                    first_seen_at=wall_time,
                    last_seen_at=wall_time,
                    total_visits=1,
                )
                write_db.add(ident)
                write_db.commit()
                self.logger.info("live_new_person_created", label=person_label)
            except Exception as e:
                self.logger.error("live_identity_create_failed",
                                  label=person_label, error=str(e))
                try: write_db.rollback()
                except Exception: pass

        # ── Create session record ─────────────────────────────────────────────
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
            self.logger.error("live_session_create_failed", error=str(e))
            try: write_db.rollback()
            except Exception: pass

        return person_label, session_id

    def _close_stale_tracks(self, wall_time: datetime, write_db):
        """Close sessions for persons not seen recently."""
        timeout  = self.settings.live_exit_timeout_seconds
        to_close = []

        with self._tracks_lock:
            for tid, track in list(self._tracks.items()):
                if (wall_time - track.last_seen).total_seconds() > timeout:
                    to_close.append((tid, track))

        for tid, track in to_close:
            self._close_session(track, wall_time, write_db)
            with self._tracks_lock:
                self._tracks.pop(tid, None)
            self.logger.info(
                "live_person_exited",
                person=track.person_label,
                duration=f"{(wall_time - track.first_seen).total_seconds():.0f}s",
            )

    def _close_all_sessions(self, wall_time: datetime, db):
        with self._tracks_lock:
            tracks = list(self._tracks.values())
        for track in tracks:
            self._close_session(track, wall_time, db)

    def _close_session(self, track: LiveTrack, exit_time: datetime, db):
        from app.storage.live_models import PersonSession
        import uuid as _uuid
        try:
            session = db.query(PersonSession).filter(
                PersonSession.id == _uuid.UUID(track.session_id)).first()
            if session:
                session.exit_time         = exit_time
                session.duration_seconds  = (exit_time - track.first_seen).total_seconds()
                session.last_state        = track.last_state
                session.best_crop_path    = track.best_crop_path
                session.is_active         = False
                db.commit()
        except Exception as e:
            self.logger.warning("live_session_close_failed", error=str(e))
            try: db.rollback()
            except Exception: pass

    def _persist_session(self, track: LiveTrack, db):
        """Periodic update to session row (rolling exit time + state)."""
        from app.storage.live_models import PersonSession
        import uuid as _uuid
        try:
            session = db.query(PersonSession).filter(
                PersonSession.id == _uuid.UUID(track.session_id)).first()
            if session:
                session.exit_time         = track.last_seen
                session.duration_seconds  = (track.last_seen - track.first_seen).total_seconds()
                session.last_state        = track.last_state
                session.best_crop_path    = track.best_crop_path
                session.last_bbox         = {
                    "x1": track.last_bbox[0], "y1": track.last_bbox[1],
                    "x2": track.last_bbox[2], "y2": track.last_bbox[3],
                }
                db.commit()
        except Exception as e:
            self.logger.debug("live_persist_failed", error=str(e))
            try: db.rollback()
            except Exception: pass

    def _update_embedding_db(self, person_label: str, crop_path: str, db):
        """Update identity embedding with new crop (non-critical, no-throw)."""
        from app.storage.live_models import PersonIdentity
        try:
            new_emb = self.reid.embed_crop(crop_path)
            if new_emb is None:
                return
            ident = db.query(PersonIdentity).filter(
                PersonIdentity.person_label == person_label).first()
            if ident and ident.embedding:
                stored  = np.array(ident.embedding, dtype=np.float32)
                blended = 0.7 * stored + 0.3 * new_emb
                n       = np.linalg.norm(blended)
                ident.embedding = (blended / n if n > 0 else blended).tolist()
                db.commit()
        except Exception:
            try: db.rollback()
            except Exception: pass

    # ── Frame annotation ──────────────────────────────────────────────────────

    def _enhance_brightness(self, frame: np.ndarray) -> np.ndarray:
        """
        Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) to brighten
        dark RTSP feeds for display. Only applied to the MJPEG stream — detection
        always uses the raw unmodified frame for accuracy.

        CLAHE works per-channel in LAB color space so colors stay natural.
        clipLimit=3.0 and tileSize=(8,8) give strong enhancement without artifacts.
        """
        try:
            lab  = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            cl   = clahe.apply(l)
            enhanced = cv2.merge((cl, a, b))
            return cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
        except Exception:
            return frame.copy()


    def _annotate_frame(self, frame: np.ndarray) -> np.ndarray:
        # Apply CLAHE brightness enhancement for display
        # Detection runs on raw frame; this only affects the MJPEG stream shown in UI
        out = self._enhance_brightness(frame)
        h, w = out.shape[:2]

        # Timestamp
        ts = datetime.utcnow().strftime("%d-%m-%Y  %H:%M:%S  UTC")
        cv2.putText(out, ts, (w - 320, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (200, 200, 200), 1, cv2.LINE_AA)

        # LIVE badge
        cv2.circle(out, (20, 20), 8, (0, 0, 220), -1)
        cv2.putText(out, "LIVE", (34, 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220, 220, 220), 1, cv2.LINE_AA)

        now     = datetime.utcnow()
        timeout = self.settings.live_exit_timeout_seconds

        with self._tracks_lock:
            tracks = list(self._tracks.values())

        for track in tracks:
            age = (now - track.last_seen).total_seconds()
            if age > timeout or track.last_bbox == (0, 0, 0, 0):
                continue

            x1, y1, x2, y2 = [int(v) for v in track.last_bbox]
            color = _color_for(track.person_label)

            # Box
            cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)

            # Label bar
            label    = f"{track.person_label}  {track.last_state.upper()}"
            (lw, lh), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
            cv2.rectangle(out, (x1, y1 - lh - 10), (x1 + lw + 8, y1), color, -1)
            cv2.putText(out, label, (x1 + 4, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)

            # Duration
            dur = (track.last_seen - track.first_seen).total_seconds()
            if dur < 60:
                dur_str = f"{int(dur)}s"
            else:
                dur_str = f"{int(dur//60)}m{int(dur%60)}s"
            cv2.putText(out, dur_str, (x1 + 4, y2 - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)

        return out

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _next_label(self) -> str:
        """Get next available person label, skipping any already in use."""
        with self._label_lock:
            while True:
                label = f"P{self._next_label_num}"
                self._next_label_num += 1
                # Skip if this label is actively in use in current session
                with self._tracks_lock:
                    active_labels = {t.person_label for t in self._tracks.values()}
                if label not in active_labels:
                    return label

    def _init_label_counter(self, db):
        """Start label counter above highest existing identity number."""
        from app.storage.live_models import PersonIdentity
        try:
            identities = db.query(PersonIdentity).all()
            nums = []
            for ident in identities:
                try:
                    nums.append(int(ident.person_label.lstrip("P")))
                except ValueError:
                    pass
            if nums:
                with self._label_lock:
                    self._next_label_num = max(nums) + 1
                self.logger.info(
                    "live_label_counter_initialized",
                    next_label=f"P{self._next_label_num}")
        except Exception as e:
            self.logger.warning("live_label_init_failed", error=str(e))

    def _save_crop(
        self, crop: np.ndarray, label: str, wall_time: datetime
    ) -> Optional[str]:
        try:
            crops_dir = self.settings.live_crops_path
            os.makedirs(crops_dir, exist_ok=True)
            ts  = wall_time.strftime("%Y%m%d_%H%M%S_%f")
            path = os.path.join(crops_dir, f"{label}_{ts}.jpg")
            cv2.imwrite(path, crop, [cv2.IMWRITE_JPEG_QUALITY, 85])
            return path
        except Exception as e:
            self.logger.debug("live_crop_save_failed", error=str(e))
            return None