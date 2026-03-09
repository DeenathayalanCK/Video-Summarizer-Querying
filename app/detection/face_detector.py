"""
face_detector.py — Lightweight face detection for surveillance keyframes.

Strategy:
  - Runs ONCE per track when the track closes (not in the real-time loop)
  - Uses OpenCV's built-in DNN face detector (Caffe ResNet-SSD model)
    which ships with OpenCV — zero extra pip installs required
  - Falls back to Haar cascade if DNN model files are unavailable
  - Detects the LARGEST face in the crop (person crop, not full frame)
  - Saves face crop to disk alongside the person crop
  - Returns FaceResult with face_crop_path, bbox, confidence, detected bool

Why at track close (not real-time):
  - People walking through a door often have face turned away initially
  - By track close we have best_crop_path — highest-confidence frame over
    the ENTIRE track duration, not just the first frame
  - Keeps real-time loop at ~3ms/frame; face detection is ~15ms
  - A single face detection per person per window is enough for surveillance

DNN model download (first run only, ~2MB):
  The Caffe ResNet-SSD face detector model files are downloaded from
  OpenCV's GitHub releases. This happens once and is cached in /data/models/.
  If download fails (no internet), falls back to Haar cascade which is
  bundled with OpenCV at no extra cost.

Output stored in TrackEvent.attributes:
  {
    "face_detected":   true | false,
    "face_crop_path":  "/data/live_crops/faces/P1_20260309_095225_face.jpg",
    "face_confidence": 0.91,
    "face_bbox":       [x1, y1, x2, y2],  # relative to person crop
    "face_method":     "dnn" | "haar" | null,
  }
"""

import os
import cv2
import numpy as np
from dataclasses import dataclass
from typing import Optional
from app.core.logging import get_logger

# ── Model URLs (OpenCV Caffe ResNet-SSD face detector) ───────────────────────
_DNN_PROTO_URL = (
    "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/"
    "face_detector/deploy.prototxt"
)
_DNN_MODEL_URL = (
    "https://github.com/opencv/opencv_3rdparty/raw/dnn_samples_face_detector_20170830/"
    "res10_300x300_ssd_iter_140000.caffemodel"
)
_MODEL_DIR      = "/data/models/face_detector"
_PROTO_PATH     = os.path.join(_MODEL_DIR, "deploy.prototxt")
_CAFFEMODEL_PATH = os.path.join(_MODEL_DIR, "res10_300x300_ssd_iter_140000.caffemodel")

# Haar cascade — bundled with OpenCV, always available
_HAAR_CASCADE   = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"

# Minimum face size relative to crop: skip tiny detections (reflections, backgrounds)
_MIN_FACE_FRACTION = 0.08   # face must be >= 8% of crop width


@dataclass
class FaceResult:
    detected:   bool
    face_crop_path: Optional[str]  = None
    confidence: float              = 0.0
    bbox:       Optional[list]     = None   # [x1, y1, x2, y2] in crop pixels
    method:     Optional[str]      = None   # "dnn" | "haar"
    error:      Optional[str]      = None


class FaceDetector:
    """
    Singleton-friendly face detector. Lazy-loads the DNN model on first use.
    Falls back to Haar cascade automatically if DNN unavailable.
    Thread-safe — only used from the track-close path (serialised per track).
    """

    _instance  = None
    _net       = None       # cv2.dnn.Net or None
    _haar      = None       # cv2.CascadeClassifier or None
    _method    = None       # "dnn" | "haar" | "none"

    def __init__(self):
        self.logger = get_logger()

    @classmethod
    def get_instance(cls) -> "FaceDetector":
        if cls._instance is None:
            cls._instance = cls()
            cls._instance._init_model()
        return cls._instance

    # ── Model initialisation ──────────────────────────────────────────────────

    def _init_model(self):
        """Try DNN first, fall back to Haar cascade."""
        if self._try_load_dnn():
            FaceDetector._method = "dnn"
            self.logger.info("face_detector_ready", method="dnn")
        elif self._try_load_haar():
            FaceDetector._method = "haar"
            self.logger.info("face_detector_ready", method="haar")
        else:
            FaceDetector._method = "none"
            self.logger.warning("face_detector_unavailable",
                                reason="both DNN and Haar failed to load")

    def _try_load_dnn(self) -> bool:
        """Download model files if needed, then load DNN net."""
        try:
            os.makedirs(_MODEL_DIR, exist_ok=True)

            # Download if not cached
            if not os.path.exists(_PROTO_PATH):
                self._download(_DNN_PROTO_URL, _PROTO_PATH)
            if not os.path.exists(_CAFFEMODEL_PATH):
                self._download(_DNN_MODEL_URL, _CAFFEMODEL_PATH)

            net = cv2.dnn.readNetFromCaffe(_PROTO_PATH, _CAFFEMODEL_PATH)
            FaceDetector._net = net
            return True
        except Exception as e:
            self.logger.warning("face_dnn_load_failed", error=str(e))
            return False

    def _try_load_haar(self) -> bool:
        try:
            cc = cv2.CascadeClassifier(_HAAR_CASCADE)
            if cc.empty():
                return False
            FaceDetector._haar = cc
            return True
        except Exception as e:
            self.logger.warning("face_haar_load_failed", error=str(e))
            return False

    @staticmethod
    def _download(url: str, dest: str):
        import urllib.request
        tmp = dest + ".tmp"
        urllib.request.urlretrieve(url, tmp)
        os.rename(tmp, dest)

    # ── Public API ────────────────────────────────────────────────────────────

    def detect(
        self,
        crop_path: str,
        save_dir: str,
        label: str,
        wall_time,
        conf_threshold: float = 0.65,
    ) -> FaceResult:
        """
        Detect the best (largest, most confident) face in a person crop.

        Args:
            crop_path:      Path to the person crop JPEG
            save_dir:       Directory to save face crop (e.g. /data/live_crops/faces)
            label:          Person label for filename (e.g. "P1")
            wall_time:      datetime for timestamped filename
            conf_threshold: Min DNN confidence to accept a detection

        Returns:
            FaceResult with detected=True/False and face_crop_path if found
        """
        if FaceDetector._method == "none":
            return FaceResult(detected=False, error="no_model")

        if not crop_path or not os.path.exists(crop_path):
            return FaceResult(detected=False, error="crop_not_found")

        try:
            img = cv2.imread(crop_path)
            if img is None or img.size == 0:
                return FaceResult(detected=False, error="crop_unreadable")

            if FaceDetector._method == "dnn":
                result = self._detect_dnn(img, conf_threshold)
            else:
                result = self._detect_haar(img)

            if not result.detected:
                return result

            # Save face crop
            face_path = self._save_face_crop(img, result.bbox, save_dir, label, wall_time)
            result.face_crop_path = face_path
            return result

        except Exception as e:
            self.logger.debug("face_detect_failed", crop=crop_path, error=str(e))
            return FaceResult(detected=False, error=str(e))

    # ── DNN detection ─────────────────────────────────────────────────────────

    def _detect_dnn(self, img: np.ndarray, conf_threshold: float) -> FaceResult:
        h, w = img.shape[:2]
        min_face_px = int(w * _MIN_FACE_FRACTION)

        blob = cv2.dnn.blobFromImage(
            cv2.resize(img, (300, 300)),
            scalefactor=1.0,
            size=(300, 300),
            mean=(104.0, 177.0, 123.0),
        )
        FaceDetector._net.setInput(blob)
        detections = FaceDetector._net.forward()  # shape (1,1,N,7)

        best_conf  = 0.0
        best_bbox  = None

        for i in range(detections.shape[2]):
            confidence = float(detections[0, 0, i, 2])
            if confidence < conf_threshold:
                continue

            x1 = int(detections[0, 0, i, 3] * w)
            y1 = int(detections[0, 0, i, 4] * h)
            x2 = int(detections[0, 0, i, 5] * w)
            y2 = int(detections[0, 0, i, 6] * h)

            # Clamp to image bounds
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            fw, fh  = x2 - x1, y2 - y1

            if fw < min_face_px or fh < min_face_px:
                continue  # too small — likely background noise

            face_area = fw * fh
            if confidence > best_conf or (
                confidence > best_conf - 0.05 and face_area > (best_bbox[2]-best_bbox[0])*(best_bbox[3]-best_bbox[1]) if best_bbox else False
            ):
                best_conf = confidence
                best_bbox = [x1, y1, x2, y2]

        if best_bbox:
            return FaceResult(
                detected=True,
                confidence=round(best_conf, 3),
                bbox=best_bbox,
                method="dnn",
            )
        return FaceResult(detected=False, method="dnn")

    # ── Haar detection ────────────────────────────────────────────────────────

    def _detect_haar(self, img: np.ndarray) -> FaceResult:
        h, w = img.shape[:2]
        min_face_px = int(w * _MIN_FACE_FRACTION)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)

        faces = FaceDetector._haar.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(min_face_px, min_face_px),
            flags=cv2.CASCADE_SCALE_IMAGE,
        )

        if not len(faces):
            return FaceResult(detected=False, method="haar")

        # Pick largest face
        best = max(faces, key=lambda f: f[2] * f[3])
        x, y, fw, fh = best
        bbox = [x, y, x + fw, y + fh]

        return FaceResult(
            detected=True,
            confidence=0.75,   # Haar doesn't produce confidence scores
            bbox=bbox,
            method="haar",
        )

    # ── Save face crop ────────────────────────────────────────────────────────

    def _save_face_crop(
        self,
        img: np.ndarray,
        bbox: list,
        save_dir: str,
        label: str,
        wall_time,
    ) -> Optional[str]:
        try:
            x1, y1, x2, y2 = bbox
            # Add 15% padding around face for better context
            h, w = img.shape[:2]
            fw, fh = x2 - x1, y2 - y1
            pad_x = int(fw * 0.15)
            pad_y = int(fh * 0.15)
            x1p = max(0, x1 - pad_x)
            y1p = max(0, y1 - pad_y)
            x2p = min(w, x2 + pad_x)
            y2p = min(h, y2 + pad_y)

            face_crop = img[y1p:y2p, x1p:x2p]
            if face_crop.size == 0:
                return None

            os.makedirs(save_dir, exist_ok=True)
            ts   = wall_time.strftime("%Y%m%d_%H%M%S")
            path = os.path.join(save_dir, f"{label}_{ts}_face.jpg")
            cv2.imwrite(path, face_crop, [cv2.IMWRITE_JPEG_QUALITY, 92])
            return path
        except Exception:
            return None