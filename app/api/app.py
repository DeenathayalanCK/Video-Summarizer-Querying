from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from contextlib import asynccontextmanager
import os

from app.api.routes import router
from app.api.live_routes import router as live_router
from app.core.logging import setup_logging, get_logger

setup_logging()
logger = get_logger()


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("api_starting")
    from app.core.config import get_settings
    settings = get_settings()
    # Create live stream tables if they don't exist
    try:
        from app.storage.database import engine
        from app.storage.live_models import PersonIdentity, PersonSession
        from sqlalchemy import inspect
        inspector = inspect(engine)
        existing = inspector.get_table_names()
        if "person_identities" not in existing or "person_sessions" not in existing:
            from app.storage.database import Base
            # Import live models so Base knows about them
            import app.storage.live_models  # noqa: F401
            Base.metadata.create_all(bind=engine, tables=[
                PersonIdentity.__table__,
                PersonSession.__table__,
            ])
            logger.info("live_tables_created")
    except Exception as e:
        logger.warning("live_tables_init_failed", error=str(e))

    # Auto-start live stream + window manager if VIDEO_INPUT_PATH is RTSP
    if settings.is_rtsp:
        try:
            from app.vision.window_manager import WindowManager
            WindowManager.get_instance().start()
            logger.info("window_manager_auto_started")
        except Exception as e:
            logger.warning("window_manager_start_failed", error=str(e))
        try:
            from app.vision.live_stream_processor import LiveStreamProcessor
            LiveStreamProcessor.get_instance().start()
            logger.info("live_stream_auto_started")
        except Exception as e:
            logger.warning("live_stream_start_failed", error=str(e))

    yield
    # Graceful shutdown: stop live stream if running
    try:
        from app.vision.window_manager import WindowManager
        WindowManager.get_instance().stop()
    except Exception:
        pass
    try:
        from app.vision.live_stream_processor import LiveStreamProcessor
        proc = LiveStreamProcessor.get_instance()
        if proc.is_running:
            proc.stop()
    except Exception:
        pass
    logger.info("api_shutdown")


app = FastAPI(
    title="VideoRAG API",
    description="Search and query video content using natural language",
    version="3.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# API routes
app.include_router(router, prefix="/api/v1")
app.include_router(live_router, prefix="/api/v1")

# Serve static UI — mount at /ui, and root redirects there
STATIC_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "static")

if os.path.exists(STATIC_DIR):
    app.mount("/ui", StaticFiles(directory=STATIC_DIR), name="static")

    @app.get("/")
    def root():
        return FileResponse(os.path.join(STATIC_DIR, "index.html"))