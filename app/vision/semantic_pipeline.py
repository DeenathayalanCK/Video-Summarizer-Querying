from datetime import datetime
from app.vision.scene_change import SceneChangeDetector
from app.captioning.ollama_client import OllamaMultimodalClient
from app.storage.database import SessionLocal
from app.storage.repository import EventRepository


class SemanticVideoProcessor:
    def __init__(self, camera_id, threshold):
        self.camera_id = camera_id
        self.scene_detector = SceneChangeDetector(threshold)
        self.captioner = OllamaMultimodalClient()

    def process_frame(self, frame):
        if self.scene_detector.is_scene_changed(frame):
            caption = self.captioner.generate_caption(frame)

            db = SessionLocal()
            try:
                repo = EventRepository(db)
                repo.save_caption(
                    camera_id=self.camera_id,
                    frame_timestamp=datetime.utcnow(),
                    caption_text=caption,
                )
            finally:
                db.close()

            return caption

        return None