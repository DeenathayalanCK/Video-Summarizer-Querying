from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from contextlib import asynccontextmanager
import os

from app.api.routes import router
from app.core.logging import setup_logging, get_logger

setup_logging()
logger = get_logger()


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("api_starting")
    yield
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

# Serve static UI — mount at /ui, and root redirects there
STATIC_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "static")

if os.path.exists(STATIC_DIR):
    app.mount("/ui", StaticFiles(directory=STATIC_DIR), name="static")

    @app.get("/")
    def root():
        return FileResponse(os.path.join(STATIC_DIR, "index.html"))