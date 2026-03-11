from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from app.core.config import get_settings
from app.storage.models import Base

settings = get_settings()

engine = create_engine(
    settings.database_url,
    pool_pre_ping=True,
    pool_recycle=1800,   # recycle connections every 30 min (avoids stale-conn drops)
    pool_timeout=30,     # raise after 30 s waiting for a free connection
    pool_size=5,
    max_overflow=10,
)

SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine,
)


def _ensure_processing_status_columns():
    ddl = [
        "ALTER TABLE processing_status ADD COLUMN IF NOT EXISTS attrs_completed BOOLEAN NOT NULL DEFAULT FALSE",
        "ALTER TABLE processing_status ADD COLUMN IF NOT EXISTS activity_completed BOOLEAN NOT NULL DEFAULT FALSE",
        "ALTER TABLE processing_status ADD COLUMN IF NOT EXISTS reembed_completed BOOLEAN NOT NULL DEFAULT FALSE",
        "ALTER TABLE processing_status ADD COLUMN IF NOT EXISTS summary_completed BOOLEAN NOT NULL DEFAULT FALSE",
    ]
    with engine.begin() as conn:
        for stmt in ddl:
            conn.execute(text(stmt))


def init_db():
    Base.metadata.create_all(bind=engine)
    _ensure_processing_status_columns()
