import time
from sqlalchemy import text
from sqlalchemy.exc import OperationalError

from app.core.logging import setup_logging, get_logger
from app.storage.database import engine, init_db
from app.vision.semantic_video_processor import SemanticVideoProcessor


MAX_RETRIES = 10
RETRY_DELAY = 3


def wait_for_db(logger):
    retries = 0
    while retries < MAX_RETRIES:
        try:
            with engine.connect() as connection:
                result = connection.execute(text("SELECT 1"))
                return result.scalar()
        except OperationalError as e:
            retries += 1
            logger.warning(
                "database_not_ready_retrying",
                attempt=retries,
                error=str(e),
            )
            time.sleep(RETRY_DELAY)

    raise Exception("Database not ready after retries")


def main():
    setup_logging()
    logger = get_logger()

    logger.info("starting_application")

    db_result = wait_for_db(logger)
    logger.info("database_connection_successful", result=db_result)

    init_db()
    logger.info("database_tables_initialized")

    logger.info("semantic_pipeline_initializing")

    processor = SemanticVideoProcessor()
    processor.run()

    logger.info("semantic_pipeline_completed")


if __name__ == "__main__":
    main()