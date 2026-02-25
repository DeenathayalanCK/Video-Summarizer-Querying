import time
import requests
from sqlalchemy import text
from sqlalchemy.exc import OperationalError

from app.core.logging import setup_logging, get_logger
from app.core.config import get_settings
from app.storage.database import engine, init_db
from app.vision.semantic_video_processor import SemanticVideoProcessor


MAX_RETRIES = 10
RETRY_DELAY = 3


def wait_for_db(logger):
    retries = 0
    while retries < MAX_RETRIES:
        try:
            with engine.connect() as conn:
                return conn.execute(text("SELECT 1")).scalar()
        except OperationalError as e:
            retries += 1
            logger.warning("database_not_ready_retrying", attempt=retries, error=str(e))
            time.sleep(RETRY_DELAY)
    raise Exception("Database not ready after retries")


def wait_for_ollama(logger):
    settings = get_settings()
    url = f"{settings.ollama_host}/api/tags"
    retries = 0
    while retries < MAX_RETRIES:
        try:
            if requests.get(url, timeout=5).status_code == 200:
                logger.info("ollama_ready", host=settings.ollama_host)
                return
        except requests.exceptions.RequestException:
            pass
        retries += 1
        logger.warning("ollama_not_ready_retrying", attempt=retries)
        time.sleep(RETRY_DELAY)
    raise Exception("Ollama not ready after retries")


def ensure_pgvector(logger):
    with engine.connect() as conn:
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        conn.commit()
    logger.info("pgvector_extension_ready")


def pull_model(logger, model_name: str):
    """Pull a model from Ollama if not already present. Idempotent."""
    settings = get_settings()
    logger.info("pulling_model", model=model_name)
    try:
        response = requests.post(
            f"{settings.ollama_host}/api/pull",
            json={"name": model_name, "stream": False},
            timeout=600,
        )
        response.raise_for_status()
        logger.info("model_ready", model=model_name)
    except Exception as e:
        logger.error("model_pull_failed", model=model_name, error=str(e))
        raise


def main():
    setup_logging()
    logger = get_logger()
    settings = get_settings()

    logger.info("starting_application")

    wait_for_db(logger)
    logger.info("database_connection_successful")

    ensure_pgvector(logger)

    init_db()
    logger.info("database_tables_initialized")

    wait_for_ollama(logger)

    # Pull all required models — idempotent, skips if already present
    pull_model(logger, settings.multimodal_model)   # llava:7b  — for captioning
    pull_model(logger, settings.embed_model)         # nomic-embed-text — for search
    pull_model(logger, settings.text_model)          # llama3.2 — for Q&A

    logger.info("semantic_pipeline_initializing")
    processor = SemanticVideoProcessor()
    processor.run()
    logger.info("semantic_pipeline_completed")


if __name__ == "__main__":
    main()