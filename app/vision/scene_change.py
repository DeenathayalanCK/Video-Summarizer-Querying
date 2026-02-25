import numpy as np
import cv2


class SceneChangeDetector:
    def __init__(self, threshold: float):
        self.threshold = threshold
        self.previous_frame = None

    def is_scene_changed(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if self.previous_frame is None:
            self.previous_frame = gray
            return True

        diff = cv2.absdiff(self.previous_frame, gray)
        mean_diff = np.mean(diff)

        self.previous_frame = gray

        return mean_diff > self.threshold