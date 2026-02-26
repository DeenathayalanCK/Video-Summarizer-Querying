import numpy as np
import cv2
from collections import deque


class SceneChangeDetector:
    """
    Dual-window scene change detector:

    SHORT-WINDOW — frame vs previous frame (catches sudden cuts/entries)
    LONG-WINDOW  — current frame top-region vs N seconds ago
                   (catches slow creep entries like a car gradually entering)

    A cooldown period prevents the same event from generating
    duplicate captures on consecutive frames.
    """

    # Low-information frame filters
    DARK_THRESHOLD = 20
    BRIGHT_THRESHOLD = 235
    UNIFORM_RATIO = 0.90
    MIN_LAPLACIAN_VARIANCE = 15.0

    # Top region fraction used for long-window comparison
    # Overhead/elevated security cameras see vehicles enter from upper frame
    TOP_REGION_FRACTION = 0.6

    def __init__(
        self,
        threshold: float,
        long_window_seconds: int = 8,
        long_window_threshold: float = 25.0,
        cooldown_seconds: int = 3,
    ):
        self.threshold = threshold
        self.long_window_seconds = long_window_seconds
        self.long_threshold = long_window_threshold
        self.cooldown_seconds = cooldown_seconds

        self._history: deque = deque(maxlen=long_window_seconds)
        self.previous_frame = None
        self._last_capture_second: float = -999.0

    def is_scene_changed(self, frame, current_second: float = 0.0) -> bool:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        h = gray.shape[0]
        gray_top = gray[:int(h * self.TOP_REGION_FRACTION), :]

        # First frame
        if self.previous_frame is None:
            self.previous_frame = gray
            self._history.append(gray_top)
            self._last_capture_second = current_second
            return not self._is_low_information(gray)

        short_diff = float(np.mean(cv2.absdiff(self.previous_frame, gray)))
        long_diff = float(np.mean(cv2.absdiff(self._history[0], gray_top)))

        self.previous_frame = gray
        self._history.append(gray_top)

        # Respect cooldown
        if (current_second - self._last_capture_second) < self.cooldown_seconds:
            return False

        changed = (
            short_diff > self.threshold or
            long_diff > self.long_threshold
        )

        if not changed:
            return False

        if self._is_low_information(gray):
            return False

        self._last_capture_second = current_second
        return True

    def _is_low_information(self, gray: np.ndarray) -> bool:
        total = gray.size
        dark = np.sum(gray < self.DARK_THRESHOLD)
        bright = np.sum(gray > self.BRIGHT_THRESHOLD)
        if (dark + bright) / total > self.UNIFORM_RATIO:
            return True
        if float(cv2.Laplacian(gray, cv2.CV_64F).var()) < self.MIN_LAPLACIAN_VARIANCE:
            return True
        return False

    def reset(self):
        """Reset state between videos."""
        self.previous_frame = None
        self._history.clear()
        self._last_capture_second = -999.0