import cv2


class FrameSampler:
    def __init__(self, video_path: str, sample_fps: int):
        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)

        if not self.cap.isOpened():
            raise RuntimeError(f"Failed to open video: {video_path}")

        self.original_fps = self.cap.get(cv2.CAP_PROP_FPS)

        if not self.original_fps or self.original_fps <= 0:
            raise RuntimeError(
                f"Invalid FPS detected for video: {video_path}"
            )

        if sample_fps <= 0:
            raise ValueError("sample_fps must be greater than 0")

        self.sample_fps = sample_fps
        self.frame_interval = max(
            int(self.original_fps / sample_fps), 1
        )
        self.frame_count = 0

    def __iter__(self):
        return self

    def __next__(self):
        while True:
            ret, frame = self.cap.read()

            if not ret:
                self._release()
                raise StopIteration

            self.frame_count += 1

            if self.frame_count % self.frame_interval == 0:
                timestamp_sec = (
                    self.cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
                )
                return frame, float(timestamp_sec)

    def _release(self):
        if self.cap and self.cap.isOpened():
            self.cap.release()

    def __del__(self):
        """Ensure the video capture is always released, even on exceptions."""
        self._release()
