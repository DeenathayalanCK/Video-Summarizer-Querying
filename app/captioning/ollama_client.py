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

    def _encode_image(self, frame) -> str:
        # FIX: resize large frames before encoding
        # Your frames are 2688x1520 (4MP) — vision models process at 336-448px
        # Sending full resolution burns CPU time without improving caption quality
        h, w = frame.shape[:2]
        max_dim = self.settings.caption_max_image_dim  # default 768
        if max(h, w) > max_dim:
            scale = max_dim / max(h, w)
            frame = cv2.resize(
                frame,
                (int(w * scale), int(h * scale)),
                interpolation=cv2.INTER_AREA,
            )
        _, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        return base64.b64encode(buffer).decode("utf-8")

    def generate_caption(self, frame) -> str:
        image_b64 = self._encode_image(frame)

        payload = {
            "model": self.model,
            "prompt": self.prompt,
            "images": [image_b64],
            "stream": False,
            "options": {
                # FIX: cap token output — prevents model writing essays
                "num_predict": self.settings.caption_max_tokens,
                "temperature": 0.1,
            },
        }

        response = requests.post(
            f"{self.base_url}/api/generate",
            json=payload,
            # FIX: use config value instead of hardcoded 700
            timeout=self.settings.caption_timeout_seconds,
        )
        response.raise_for_status()
        return response.json()["response"]