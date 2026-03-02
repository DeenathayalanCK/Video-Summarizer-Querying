from pydantic_settings import BaseSettings
from pydantic import Field
from functools import lru_cache


class Settings(BaseSettings):
    app_env: str = Field(..., alias="APP_ENV")

    db_host: str = Field(..., alias="DB_HOST")
    db_port: int = Field(..., alias="DB_PORT")
    db_name: str = Field(..., alias="DB_NAME")
    db_user: str = Field(..., alias="DB_USER")
    db_password: str = Field(..., alias="DB_PASSWORD")

    log_level: str = Field(..., alias="LOG_LEVEL")
    service_name: str = Field(..., alias="SERVICE_NAME")

    ollama_host: str = Field(..., alias="OLLAMA_HOST")

    # Vision model — used for crop attribute extraction (Phase 6B)
    # and optional whole-frame captioning of anomaly frames
    multimodal_model: str = Field(..., alias="MULTIMODAL_MODEL")

    # Text model — for Q&A and summarization
    text_model: str = Field(..., alias="TEXT_MODEL")

    # Embedding model — for semantic search (768-dim)
    embed_model: str = Field(..., alias="EMBED_MODEL")

    frame_sample_fps: int = Field(..., alias="FRAME_SAMPLE_FPS")

    # Scene change detection (still used as a secondary signal)
    scene_change_threshold: float = Field(..., alias="SCENE_CHANGE_THRESHOLD")
    scene_long_window_seconds: int = Field(8, alias="SCENE_LONG_WINDOW_SECONDS")
    scene_long_window_threshold: float = Field(25.0, alias="SCENE_LONG_WINDOW_THRESHOLD")
    scene_cooldown_seconds: int = Field(3, alias="SCENE_COOLDOWN_SECONDS")

    # Captioning performance
    caption_timeout_seconds: int = Field(300, alias="CAPTION_TIMEOUT_SECONDS")
    caption_max_tokens: int = Field(512, alias="CAPTION_MAX_TOKENS")
    caption_max_image_dim: int = Field(768, alias="CAPTION_MAX_IMAGE_DIM")

    # ── Phase 6A: YOLO detection settings ────────────────────────────────────

    # YOLO model name — ultralytics auto-downloads on first run
    # Options (CPU-friendly, fastest to slowest):
    #   yolov8n.pt  — nano,   ~80ms/frame,  best for CPU
    #   yolov8s.pt  — small,  ~150ms/frame, better accuracy
    #   yolov8m.pt  — medium, ~300ms/frame, for powerful machines
    yolo_model: str = Field("yolov8n.pt", alias="YOLO_MODEL")

    # Optional: absolute path to a local .pt file (skips auto-download)
    # Leave empty to use auto-download
    yolo_model_path: str = Field("", alias="YOLO_MODEL_PATH")

    # Minimum detection confidence to keep a bounding box
    # Lower = more detections but more false positives
    # 0.35 is a good balance for surveillance footage
    yolo_confidence_threshold: float = Field(0.35, alias="YOLO_CONFIDENCE_THRESHOLD")

    # ── Phase 6A: Tracking settings ───────────────────────────────────────────

    # Minimum seconds an object must be present to generate a dwell event
    dwell_threshold_seconds: float = Field(10.0, alias="DWELL_THRESHOLD_SECONDS")

    # If an object disappears for > this many seconds, generate an exit event
    exit_gap_seconds: float = Field(3.0, alias="EXIT_GAP_SECONDS")

    # Minimum confidence for a detection to save a crop to disk
    # Crops are used for Phase 6B attribute extraction
    crop_min_confidence: float = Field(0.5, alias="CROP_MIN_CONFIDENCE")

    # ── Paths ─────────────────────────────────────────────────────────────────

    video_input_path: str = Field(..., alias="VIDEO_INPUT_PATH")
    camera_id: str = Field(..., alias="CAMERA_ID")

    @property
    def database_url(self) -> str:
        return (
            f"postgresql+psycopg2://{self.db_user}:"
            f"{self.db_password}@{self.db_host}:"
            f"{self.db_port}/{self.db_name}"
        )

    class Config:
        env_file = ".env"
        case_sensitive = True


@lru_cache
def get_settings() -> Settings:
    return Settings()