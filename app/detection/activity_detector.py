"""
activity_detector.py — Two-stage activity detection for live track events.

STAGE 1: Real-time (runs every N frames on expanded context crop, ~5ms)
  YOLO on a 3× expanded person bbox — includes the desk workspace, not just torso.
  This ensures laptop, phone, keyboard, monitor are visible to YOLO.
  Stores detected objects in track.objects_nearby + TrackEvent.attributes.
  Reuses the SAME ObjectDetector YOLO model — no second model loaded.

STAGE 2: Post-window (runs async after window closes)
  minicpm-v on best crop → natural one-liner activity caption per track.
  Stored in TrackEvent.attributes["activity_caption"] + appended to rag_text.

STAGE 3: Every-minute activity snapshot (new)
  If a person has phone/laptop/screen detected, log a timestamped snapshot
  every 60 wall-clock seconds as a TrackEvent with event_type="activity_snapshot".
  This answers queries like "who was using a laptop at 10:15?" precisely.

YOLO COCO classes relevant to office/person activity:
  41=cup, 62=tv_monitor, 63=laptop, 64=mouse, 65=remote,
  66=keyboard, 67=cell_phone, 72=book, 73=clock
"""

import os
import math
from datetime import datetime
from typing import Optional

import numpy as np

from app.core.logging import get_logger

# COCO class IDs → activity-relevant labels
ACTIVITY_OBJECTS = {
    41:  "cup",
    62:  "tv_monitor",
    63:  "laptop",
    64:  "mouse",
    65:  "remote",
    66:  "keyboard",
    67:  "cell_phone",
    72:  "book",
    73:  "clock",
}

# Activities that warrant every-minute snapshots
SNAPSHOT_ACTIVITIES = {
    "working on laptop",
    "using phone",
    "working on laptop, using phone",
    "reading / taking notes",
    "looking at screen",
}


def infer_activity(objects: list) -> str:
    objs = set(objects)
    if not objs:
        return "present"
    has_phone    = "cell_phone" in objs
    has_laptop   = "laptop" in objs
    has_keyboard = "keyboard" in objs
    has_book     = "book" in objs
    has_cup      = "cup" in objs
    has_monitor  = "tv_monitor" in objs

    if has_phone and has_laptop:
        return "working on laptop, using phone"
    if has_laptop or (has_keyboard and has_monitor):
        return "working on laptop"
    if has_phone:
        return "using phone"
    if has_book:
        return "reading / taking notes"
    if has_monitor:
        return "looking at screen"
    if has_cup:
        return "at desk"
    return "present"


class ActivityDetector:
    """
    Stage 1: YOLO activity detection on an EXPANDED person context crop.

    Key insight: a tight person bbox (150×300px) shows only torso/head —
    YOLO cannot see the laptop on the desk or the phone at waist level.
    Expanding by 3× (or to at least 400×400) includes the workspace.

    Reuses the detector already loaded by LiveStreamProcessor — zero extra RAM.
    """

    def __init__(self):
        self.logger  = get_logger()
        self._model  = None   # lazily shared with ObjectDetector

    def detect_on_context(
        self,
        frame: np.ndarray,
        bbox: tuple,               # (x1, y1, x2, y2) in pixel coords
        expand: float = 2.5,       # expand factor around person bbox
        confidence: float = 0.30,  # lower than default — small objects in crops
    ) -> tuple:
        """
        Expand the person bbox by `expand` factor, crop from full frame,
        run YOLO restricted to ACTIVITY_OBJECTS classes.

        Returns (object_list, activity_hint).
        """
        if frame is None or frame.size == 0 or bbox == (0, 0, 0, 0):
            return [], "present"

        try:
            h, w = frame.shape[:2]
            x1, y1, x2, y2 = bbox
            bw = x2 - x1
            bh = y2 - y1

            # Expand symmetrically — include desk area above/below/sides
            pad_x = bw * (expand - 1) / 2
            pad_y = bh * (expand - 1) / 2
            ex1 = max(0,  int(x1 - pad_x))
            ey1 = max(0,  int(y1 - pad_y))
            ex2 = min(w,  int(x2 + pad_x))
            ey2 = min(h,  int(y2 + pad_y))

            context_crop = frame[ey1:ey2, ex1:ex2]
            if context_crop.size == 0:
                return [], "present"

            # Lazy-load or reuse model
            if self._model is None:
                from ultralytics import YOLO
                from app.core.config import get_settings
                s = get_settings()
                self._model = YOLO(s.yolo_model_path or s.yolo_model)

            results = self._model(
                context_crop,
                verbose=False,
                conf=confidence,
                classes=list(ACTIVITY_OBJECTS.keys()),
            )

            detected = []
            if results and results[0].boxes is not None:
                for cls_id in results[0].boxes.cls.tolist():
                    label = ACTIVITY_OBJECTS.get(int(cls_id))
                    if label and label not in detected:
                        detected.append(label)

            activity = infer_activity(detected)
            return detected, activity

        except Exception as e:
            self.logger.debug("activity_detect_failed", error=str(e))
            return [], "present"


class ActivityCaptioner:
    """
    Stage 2: minicpm-v one-liner activity caption per track, post-window.
    """

    _PROMPT = (
        "Look at this image. In ONE SHORT SENTENCE (max 12 words), describe "
        "what the person is doing. Focus on activity, not appearance. "
        "Examples: 'Person typing on a laptop', 'Person talking on a phone', "
        "'Person reading documents', 'Person sitting at a desk'. "
        "Only describe what is clearly visible. Reply with the sentence only."
    )

    def __init__(self, ollama_host: str, model: str):
        self.ollama_host = ollama_host
        self.model       = model
        self.logger      = get_logger()

    def caption(self, crop_path: str) -> Optional[str]:
        if not crop_path or not os.path.exists(crop_path):
            return None
        try:
            import base64, requests
            with open(crop_path, "rb") as f:
                img_b64 = base64.b64encode(f.read()).decode()
            resp = requests.post(
                f"{self.ollama_host}/api/generate",
                json={
                    "model":  self.model,
                    "prompt": self._PROMPT,
                    "images": [img_b64],
                    "stream": False,
                    "options": {"num_predict": 30, "temperature": 0.1},
                },
                timeout=30,
            )
            if resp.status_code == 200:
                raw     = resp.json().get("response", "").strip()
                caption = raw.split(".")[0].strip().strip('"\'')
                if len(caption) > 5:
                    return caption
        except Exception as e:
            self.logger.debug("activity_caption_failed", crop=crop_path, error=str(e))
        return None


# ── Stage 3: Every-minute activity snapshot ───────────────────────────────────

def write_activity_snapshot(track, window_key: str, wall_time: datetime, db):
    """
    Write a TrackEvent(event_type="activity_snapshot") for a person currently
    detected doing a notable activity (phone/laptop/screen).

    Called by LiveStreamProcessor every 60s per active track when the track
    has a non-trivial activity_hint.  These rows power queries like:
      "Who was using a laptop between 10:00 and 11:00?"
    """
    from app.storage.models import TrackEvent
    from app.core.config import get_settings
    import time as _time
    if track.activity_hint not in SNAPSHOT_ACTIVITIES:
        return
    try:
        epoch_sec = _time.mktime(wall_time.timetuple())
        dur = (wall_time - track.first_seen).total_seconds()
        ev = TrackEvent(
            video_filename=window_key,
            camera_id=get_settings().camera_id,
            track_id=track.track_id,
            object_class="person",
            event_type="activity_snapshot",
            first_seen_second=epoch_sec,
            last_seen_second=epoch_sec,
            duration_seconds=round(dur, 1),
            best_frame_second=epoch_sec,
            best_crop_path=track.best_crop_path,
            best_confidence=track.best_confidence,
            rag_text=(
                f"{track.person_label} — {track.activity_hint} "
                f"at {wall_time.strftime('%H:%M:%S')} "
                f"(present {int(dur)}s so far). "
                f"Objects nearby: {', '.join(track.objects_nearby) or 'none'}."
            ),
            attributes={
                "person_label":   track.person_label,
                "activity":       track.activity_hint,
                "objects_nearby": track.objects_nearby,
                "snapshot_time":  wall_time.isoformat(),
                "duration_so_far": round(dur, 1),
                "live":           True,
            },
        )
        db.add(ev)
        db.commit()
        get_logger().info(
            "activity_snapshot_written",
            person=track.person_label,
            activity=track.activity_hint,
            objects=track.objects_nearby,
            time=wall_time.strftime("%H:%M:%S"),
        )
    except Exception as e:
        get_logger().debug("activity_snapshot_failed", error=str(e))
        try: db.rollback()
        except Exception: pass


def run_activity_captions_for_window(video_filename: str, db) -> int:
    """
    Post-window: minicpm-v caption for every entry TrackEvent that has a crop.
    Returns count of captions written.
    """
    from app.storage.models import TrackEvent
    from app.core.config import get_settings

    s         = get_settings()
    captioner = ActivityCaptioner(ollama_host=s.ollama_host, model=s.multimodal_model)
    logger    = get_logger()
    events    = (
        db.query(TrackEvent)
        .filter(
            TrackEvent.video_filename == video_filename,
            TrackEvent.event_type     == "entry",
        )
        .all()
    )
    captioned = 0
    for ev in events:
        if not ev.best_crop_path or not os.path.exists(ev.best_crop_path):
            continue
        if ev.attributes and ev.attributes.get("activity_caption"):
            continue   # already done
        caption = captioner.caption(ev.best_crop_path)
        if not caption:
            continue
        attrs = dict(ev.attributes or {})
        attrs["activity_caption"] = caption
        ev.attributes = attrs

        person_label = attrs.get("person_label", f"track#{ev.track_id}")
        dur = ev.duration_seconds or 0
        objects_str = ", ".join(attrs.get("objects_nearby") or [])
        ev.rag_text = (
            f"{person_label} — {caption}. "
            f"Present {int(dur)}s in window {video_filename}."
            + (f" Objects: {objects_str}." if objects_str else "")
        )
        try:
            db.commit()
            captioned += 1
            logger.info("activity_captioned",
                        track=ev.track_id, caption=caption, window=video_filename)
        except Exception as e:
            logger.warning("activity_caption_save_failed", error=str(e))
            try: db.rollback()
            except Exception: pass

    return captioned