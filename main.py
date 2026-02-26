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
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            with engine.connect() as conn:
                return conn.execute(text("SELECT 1")).scalar()
        except OperationalError as e:
            logger.warning("database_not_ready_retrying", attempt=attempt, error=str(e))
            time.sleep(RETRY_DELAY)
    raise RuntimeError("Database not ready after retries")


def wait_for_ollama(logger):
    settings = get_settings()
    url = f"{settings.ollama_host}/api/tags"
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            if requests.get(url, timeout=5).status_code == 200:
                logger.info("ollama_ready", host=settings.ollama_host)
                return
        except requests.exceptions.RequestException:
            pass
        logger.warning("ollama_not_ready_retrying", attempt=attempt)
        time.sleep(RETRY_DELAY)
    raise RuntimeError("Ollama not ready after retries")


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


def recover_stale_jobs(logger):
    """
    On startup, any video stuck in 'running' state from a previous crashed
    run is marked 'failed' so it gets reprocessed this run.
    Without this, a crashed pipeline would permanently skip those videos.
    """
    from app.storage.database import SessionLocal
    from app.storage.models import ProcessingStatus
    from datetime import datetime

    db = SessionLocal()
    try:
        stale = (
            db.query(ProcessingStatus)
            .filter(ProcessingStatus.status == "running")
            .all()
        )
        for row in stale:
            row.status = "failed"
            row.last_error = "Recovered from stale running state on startup"
            row.updated_at = datetime.utcnow()
            logger.warning("stale_job_recovered", video=row.video_filename)
        if stale:
            db.commit()
            logger.info("stale_jobs_recovered", count=len(stale))
    finally:
        db.close()


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

    # Pull all required models (idempotent)
    pull_model(logger, settings.multimodal_model)
    pull_model(logger, settings.embed_model)
    pull_model(logger, settings.text_model)

    # Recover any videos stuck in 'running' from a previous crash
    recover_stale_jobs(logger)

    logger.info("semantic_pipeline_initializing")
    processor = SemanticVideoProcessor()
    processor.run()
    logger.info("semantic_pipeline_completed")


if __name__ == "__main__":
    main()