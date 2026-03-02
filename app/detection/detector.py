import os
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from ultralytics import YOLO

from app.core.config import get_settings
from app.core.logging import get_logger


# YOLO class IDs we care about for security/surveillance
# Full COCO class list has 80 classes — we only want these
SURVEILLANCE_CLASSES = {
    0:  "person",
    1:  "bicycle",
    2:  "car",
    3:  "motorcycle",
    5:  "bus",
    7:  "truck",
    # Future: 16=dog, 24=backpack, 26=handbag, 28=suitcase
}


@dataclass
class Detection:
    """
    Single object detection result from one frame.
    Bounding box coordinates are in pixel space (absolute, not normalized).
    Normalization to 0-1 happens when saving to DB.
    """
    object_class: str           # "person", "car", "truck", etc.
    confidence: float           # 0.0 - 1.0
    x1: float                   # left
    y1: float                   # top
    x2: float                   # right
    y2: float                   # bottom
    track_id: Optional[int] = None   # assigned by ByteTracker after detection


@dataclass
class FrameDetections:
    """All detections for a single frame."""
    frame_second: float
    frame_width: int
    frame_height: int
    detections: list[Detection] = field(default_factory=list)

    @property
    def has_objects(self) -> bool:
        return len(self.detections) > 0

    @property
    def person_count(self) -> int:
        return sum(1 for d in self.detections if d.object_class == "person")

    @property
    def vehicle_count(self) -> int:
        vehicle_classes = {"car", "truck", "bus", "motorcycle", "bicycle"}
        return sum(1 for d in self.detections if d.object_class in vehicle_classes)


class ObjectDetector:
    """
    Wraps YOLOv8 for surveillance object detection.

    Runs inference on every sampled frame (not just scene changes).
    YOLO on CPU takes ~80ms/frame (yolov8n) which is fast enough for 1 FPS
    batch processing of recorded video.

    Model weights are downloaded automatically on first run to:
      ~/.config/Ultralytics/  (or override via YOLO_MODEL_PATH env var)
    """

    def __init__(self):
        self.settings = get_settings()
        self.logger = get_logger()
        self._model: Optional[YOLO] = None

    def _load_model(self):
        """Lazy-load YOLO model — only loads when first detection is requested."""
        if self._model is not None:
            return

        model_name = self.settings.yolo_model
        model_path = self.settings.yolo_model_path

        # Use explicit path if configured, otherwise let ultralytics auto-download
        target = model_path if (model_path and os.path.exists(model_path)) else model_name

        self.logger.info("yolo_model_loading", model=target)
        self._model = YOLO(target)

        # Force CPU — important for systems without CUDA
        # ultralytics auto-detects but we want to be explicit
        self.logger.info("yolo_model_loaded", model=target, device="cpu")

    def warmup(self):
        """
        Load model weights and run a dummy inference pass.
        Call once at pipeline startup to pay the loading cost upfront,
        not on the first real video frame.
        """
        self._load_model()
        self.logger.info("yolo_warmup_starting")
        dummy = np.zeros((64, 64, 3), dtype=np.uint8)
        self._model.predict(
            dummy,
            device="cpu",
            verbose=False,
            classes=list(SURVEILLANCE_CLASSES.keys()),
        )
        self.logger.info("yolo_warmup_complete")

    def detect(self, frame: np.ndarray, frame_second: float) -> FrameDetections:
        """
        Run YOLO detection on a single frame.
        Returns FrameDetections with all surveillance-relevant objects found.

        Args:
            frame:        BGR numpy array (standard OpenCV format)
            frame_second: timestamp in seconds within the video

        Returns:
            FrameDetections with detection list (may be empty)
        """
        self._load_model()

        h, w = frame.shape[:2]
        result = FrameDetections(
            frame_second=frame_second,
            frame_width=w,
            frame_height=h,
        )

        # Run inference — no tracking here, tracking happens in tracker.py
        predictions = self._model.predict(
            frame,
            device="cpu",
            verbose=False,
            conf=self.settings.yolo_confidence_threshold,
            classes=list(SURVEILLANCE_CLASSES.keys()),
        )

        if not predictions or predictions[0].boxes is None:
            return result

        boxes = predictions[0].boxes
        for i in range(len(boxes)):
            cls_id = int(boxes.cls[i].item())
            if cls_id not in SURVEILLANCE_CLASSES:
                continue

            conf = float(boxes.conf[i].item())
            x1, y1, x2, y2 = boxes.xyxy[i].tolist()

            result.detections.append(Detection(
                object_class=SURVEILLANCE_CLASSES[cls_id],
                confidence=conf,
                x1=x1, y1=y1, x2=x2, y2=y2,
            ))

        return result

    def detect_with_tracking(self, frame: np.ndarray, frame_second: float) -> FrameDetections:
        """
        Run YOLO + ByteTrack in a single call (ultralytics built-in tracker).
        Returns detections with track_ids assigned.

        Uses ByteTrack which is bundled with ultralytics — no extra dependencies.
        Tracker state is maintained internally across calls (per ObjectDetector instance).
        Call reset_tracker() between videos.
        """
        self._load_model()

        h, w = frame.shape[:2]
        result = FrameDetections(
            frame_second=frame_second,
            frame_width=w,
            frame_height=h,
        )

        predictions = self._model.track(
            frame,
            device="cpu",
            verbose=False,
            conf=self.settings.yolo_confidence_threshold,
            classes=list(SURVEILLANCE_CLASSES.keys()),
            tracker="bytetrack.yaml",  # bundled with ultralytics
            persist=True,              # maintains track state across frames
        )

        if not predictions or predictions[0].boxes is None:
            return result

        boxes = predictions[0].boxes

        for i in range(len(boxes)):
            cls_id = int(boxes.cls[i].item())
            if cls_id not in SURVEILLANCE_CLASSES:
                continue

            conf = float(boxes.conf[i].item())
            x1, y1, x2, y2 = boxes.xyxy[i].tolist()

            # ByteTrack assigns track IDs — may be None if tracking lost this box
            track_id = None
            if boxes.id is not None:
                track_id = int(boxes.id[i].item())

            result.detections.append(Detection(
                object_class=SURVEILLANCE_CLASSES[cls_id],
                confidence=conf,
                x1=x1, y1=y1, x2=x2, y2=y2,
                track_id=track_id,
            ))

        return result

    def reset_tracker(self):
        """
        Reset ByteTrack state between videos.
        Without this, track IDs from video1 would bleed into video2.
        """
        if self._model is not None:
            # ultralytics resets tracker by calling predict with reset flag
            self._model.predictor = None
            self.logger.info("tracker_reset")