import requests
import base64
import cv2

from app.core.config import get_settings

from app.prompts.caption_prompt import CAPTION_PROMPT

class OllamaMultimodalClient:
    def __init__(self):
        self.settings = get_settings()
        self.base_url = self.settings.ollama_host
        self.model = self.settings.multimodal_model
        self.prompt = CAPTION_PROMPT

    def _encode_image(self, frame):
        _, buffer = cv2.imencode(".jpg", frame)
        return base64.b64encode(buffer).decode("utf-8")

    def generate_caption(self, frame) -> str:
        image_b64 = self._encode_image(frame)

        payload = {
            "model": self.model,
            "prompt": self.prompt,
            "images": [image_b64],
            "stream": False,
        }

        response = requests.post(
            f"{self.base_url}/api/generate",
            json=payload,
            timeout=700,
        )

        response.raise_for_status()

        return response.json()["response"]