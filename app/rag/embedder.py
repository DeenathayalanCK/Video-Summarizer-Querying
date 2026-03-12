import time
import requests
from app.core.config import get_settings
from app.core.logging import get_logger
from app.core.ollama_logger import log_call


class OllamaEmbedder:
    """
    Generates text embeddings via Ollama.
    Model is configured via EMBED_MODEL in .env (default: nomic-embed-text, 768-dim).
    """

    def __init__(self):
        self.settings = get_settings()
        self.logger = get_logger()
        self.base_url = self.settings.ollama_host
        self.model = self.settings.embed_model

    def embed(self, text: str) -> list[float]:
        payload = {
            "model": self.model,
            "prompt": text,
            "options": {"num_ctx": 2048},
        }
        last_exc = None
        for _attempt in range(3):
            t0 = time.monotonic()
            try:
                response = requests.post(
                    f"{self.base_url}/api/embeddings",
                    json=payload,
                    timeout=(5, 60),
                )
                response.raise_for_status()
                result = response.json()["embedding"]
                log_call(
                    call_type="embed",
                    model=self.model,
                    prompt=text[:400],
                    response=f"[{len(result)}-dim vector]",
                    elapsed_ms=(time.monotonic() - t0) * 1000,
                    status="ok",
                )
                return result
            except requests.exceptions.Timeout as e:
                log_call(
                    call_type="embed", model=self.model, prompt=text[:400],
                    elapsed_ms=(time.monotonic() - t0) * 1000,
                    status="timeout", error=str(e),
                )
                last_exc = e
                self.logger.warning("embed_timeout", attempt=_attempt + 1)
            except requests.exceptions.RequestException as e:
                log_call(
                    call_type="embed", model=self.model, prompt=text[:400],
                    elapsed_ms=(time.monotonic() - t0) * 1000,
                    status="error", error=str(e),
                )
                last_exc = e
                self.logger.warning("embed_request_error",
                                    attempt=_attempt + 1, error=str(e))
        self.logger.error("embed_failed_all_attempts", error=str(last_exc))
        return None

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [self.embed(t) for t in texts]