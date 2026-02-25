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
    """Enable pgvector extension — must run before any vector columns are used."""
    with engine.connect() as conn:
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        conn.commit()
    logger.info("pgvector_extension_ready")


def pull_embedding_model(logger):
    """
    Pull nomic-embed-text if not already present.
    Ollama pull is idempotent — safe to call every startup.
    """
    settings = get_settings()
    logger.info("pulling_embedding_model", model="nomic-embed-text")
    try:
        response = requests.post(
            f"{settings.ollama_host}/api/pull",
            json={"name": "nomic-embed-text", "stream": False},
            timeout=300,
        )
        response.raise_for_status()
        logger.info("embedding_model_ready", model="nomic-embed-text")
    except Exception as e:
        logger.error("embedding_model_pull_failed", error=str(e))
        raise


def main():
    setup_logging()
    logger = get_logger()

    logger.info("starting_application")

    wait_for_db(logger)
    logger.info("database_connection_successful")

    ensure_pgvector(logger)

    init_db()
    logger.info("database_tables_initialized")

    wait_for_ollama(logger)
    pull_embedding_model(logger)

    logger.info("semantic_pipeline_initializing")
    processor = SemanticVideoProcessor()
    processor.run()
    logger.info("semantic_pipeline_completed")


if __name__ == "__main__":
    main()