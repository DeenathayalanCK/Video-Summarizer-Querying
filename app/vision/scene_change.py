import numpy as np
import cv2


class SceneChangeDetector:
    def __init__(self, threshold: float):
        self.threshold = threshold
        self.previous_frame = None

    def is_scene_changed(self, frame) -> bool:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if self.previous_frame is None:
            self.previous_frame = gray
            return True

        diff = cv2.absdiff(self.previous_frame, gray)
        mean_diff = float(np.mean(diff))

        self.previous_frame = gray

        return mean_diff > self.threshold

    def reset(self):
        """Reset detector state between videos so each video starts fresh."""
        self.previous_frame = None
