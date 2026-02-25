import requests
from app.core.config import get_settings
from app.core.logging import get_logger

EMBEDDING_MODEL = "nomic-embed-text"


class OllamaEmbedder:
    """
    Generates text embeddings via Ollama's /api/embeddings endpoint.
    Uses nomic-embed-text (768-dim) — fast, local, no API key needed.
    """

    def __init__(self):
        self.settings = get_settings()
        self.logger = get_logger()
        self.base_url = self.settings.ollama_host
        self.model = EMBEDDING_MODEL

    def embed(self, text: str) -> list[float]:
        payload = {
            "model": self.model,
            "prompt": text,
        }

        response = requests.post(
            f"{self.base_url}/api/embeddings",
            json=payload,
            timeout=60,
        )
        response.raise_for_status()

        return response.json()["embedding"]

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed multiple texts — sequential since Ollama has no batch endpoint."""
        return [self.embed(t) for t in texts]