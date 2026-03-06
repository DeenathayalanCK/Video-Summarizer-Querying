"""
activity_detector.py — Two-stage activity detection for live track events.

STAGE 1: Real-time (runs every detection frame, <5ms)
  YOLO on person crop → detects nearby objects (phone, laptop, keyboard, cup...)
  Stores detected object classes in TrackEvent.attributes["objects_nearby"]
  Uses the SAME YOLO model already loaded — just filters to object classes.

STAGE 2: Post-window (runs async after window closes, ~10-20s per track)
  minicpm-v on best crop → generates one natural-language activity caption
  e.g. "Person using a laptop and talking on a phone"
       "Person appears to be writing or taking notes"
       "Person sitting and looking at a monitor"
  Stored in TrackEvent.attributes["activity_caption"] and appended to rag_text
  for Ask/Search to query against.

YOLO COCO classes relevant to office/person activity:
  41=cup, 42=fork, 43=knife, 44=spoon, 45=bowl
  56=chair, 57=couch, 58=potted plant, 59=bed
  62=tv, 63=laptop, 64=mouse, 65=remote, 66=keyboard, 67=cell phone
  72=book, 73=clock, 74=vase, 76=scissors, 84=book

Activity inference rules (from object combinations):
  laptop + keyboard           → "working on laptop"
  cell phone near face        → "on phone call"
  cell phone + not near face  → "using phone"
  laptop + cell phone         → "working and using phone"
  book/notebook               → "reading or taking notes"
  cup/mug                     → "having a drink"
  tv/monitor only             → "watching screen"
  no objects                  → "present"  (neutral)
"""

import os
import time
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
    84:  "book",
}

# Activity inference from detected objects
def infer_activity(objects: list[str], person_bbox: tuple = None,
                   phone_bbox: tuple = None) -> str:
    """
    Convert a list of nearby objects into a human-readable activity hint.
    Used for real-time labelling — not as rich as minicpm-v but instant.
    """
    objs = set(objects)
    if not objs:
        return "present"

    has_phone    = "cell_phone" in objs
    has_laptop   = "laptop" in objs
    has_keyboard = "keyboard" in objs
    has_book     = "book" in objs
    has_cup      = "cup" in objs
    has_monitor  = "tv_monitor" in objs

    # Multi-object inference
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
    Stage 1: YOLO-based object detection on person crops.
    Reuses the already-loaded YOLO model — zero extra memory.
    """

    def __init__(self):
        self.logger = get_logger()
        self._model = None   # set by caller to reuse existing ObjectDetector model

    def detect_objects_in_crop(
        self,
        crop: np.ndarray,
        confidence_threshold: float = 0.35,
    ) -> tuple[list[str], str]:
        """
        Run YOLO on a person crop to find nearby objects.
        Returns (object_list, activity_hint).
        Fails silently — returns ([], "present") on any error.
        """
        if crop is None or crop.size == 0:
            return [], "present"

        try:
            from ultralytics import YOLO
            if self._model is None:
                from app.core.config import get_settings
                s = get_settings()
                model_path = s.yolo_model_path or s.yolo_model
                self._model = YOLO(model_path)

            results = self._model(
                crop,
                verbose=False,
                conf=confidence_threshold,
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
    Stage 2: minicpm-v caption for a person crop.
    Runs post-window in background — same infrastructure as Phase 6B.
    """

    _PROMPT = (
        "Look at this person. In ONE SHORT SENTENCE (max 12 words), describe "
        "what they are doing. Focus on their activity, not appearance. "
        "Examples: 'Person typing on a laptop', 'Person talking on a phone', "
        "'Person reading documents', 'Person sitting at a desk'. "
        "Only describe what is clearly visible. Reply with the sentence only."
    )

    def __init__(self, ollama_host: str, model: str):
        self.ollama_host = ollama_host
        self.model       = model
        self.logger      = get_logger()

    def caption(self, crop_path: str) -> Optional[str]:
        """
        Generate a one-liner activity caption for a person crop.
        Returns None if crop missing or model unavailable.
        Times out after 30s.
        """
        if not crop_path or not os.path.exists(crop_path):
            return None

        try:
            import base64, requests
            with open(crop_path, "rb") as f:
                img_b64 = base64.b64encode(f.read()).decode()

            payload = {
                "model":  self.model,
                "prompt": self._PROMPT,
                "images": [img_b64],
                "stream": False,
                "options": {"num_predict": 30, "temperature": 0.1},
            }
            resp = requests.post(
                f"{self.ollama_host}/api/generate",
                json=payload,
                timeout=30,
            )
            if resp.status_code == 200:
                raw = resp.json().get("response", "").strip()
                # Truncate to first sentence, strip quotes
                caption = raw.split(".")[0].strip().strip('"\'')
                if len(caption) > 5:
                    return caption
        except Exception as e:
            self.logger.debug("activity_caption_failed",
                              crop=crop_path, error=str(e))
        return None


def run_activity_captions_for_window(video_filename: str, db) -> int:
    """
    Post-window pass: for each TrackEvent in this window that has a crop,
    generate an activity caption with minicpm-v and store it in attributes.
    Also updates rag_text to include the caption for Ask/Search.
    Returns count of tracks captioned.
    """
    from app.storage.models import TrackEvent
    from app.core.config import get_settings

    s = get_settings()
    captioner = ActivityCaptioner(
        ollama_host=s.ollama_host,
        model=s.multimodal_model,
    )
    logger = get_logger()

    events = (
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

        # Skip if already captioned
        if ev.attributes and ev.attributes.get("activity_caption"):
            continue

        caption = captioner.caption(ev.best_crop_path)
        if not caption:
            continue

        # Update attributes
        attrs = dict(ev.attributes or {})
        attrs["activity_caption"] = caption
        ev.attributes = attrs

        # Append caption to rag_text so Ask/Search can find it
        person_label = attrs.get("person_label", f"track#{ev.track_id}")
        dur = ev.duration_seconds or 0
        ev.rag_text = (
            f"{person_label} — {caption}. "
            f"Present for {int(dur)}s "
            f"({ev.first_seen_second:.0f}s to {ev.last_seen_second:.0f}s "
            f"in window {video_filename})."
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