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

    # Dedicated summary model: keep separate from QA so it can be tuned safely.
    summary_model: str = Field("llama3.2:latest", alias="SUMMARY_MODEL")
    summary_num_ctx: int = Field(2048, alias="SUMMARY_NUM_CTX")
    summary_max_tokens: int = Field(300, alias="SUMMARY_MAX_TOKENS")
    summary_track_limit: int = Field(6, alias="SUMMARY_TRACK_LIMIT")

    # Ask / summary generation timeouts (seconds)
    ask_timeout_seconds: int = Field(480, alias="ASK_TIMEOUT_SECONDS")
    # Max seconds Ask will wait for the Ollama semaphore before returning busy error
    ask_lane_timeout_seconds: int = Field(60, alias="ASK_LANE_TIMEOUT_SECONDS")
    summary_timeout_seconds: int = Field(420, alias="SUMMARY_TIMEOUT_SECONDS")

    # Retrieval breadth for Ask path (lower values reduce context fanout/latency)
    qa_top_k_temporal: int = Field(6, alias="QA_TOP_K_TEMPORAL")
    qa_top_k_object: int = Field(6, alias="QA_TOP_K_OBJECT")

    # Auto-summarize control.
    # true  → summary is generated automatically at end of each window/video (old behaviour).
    # false → summary is skipped during pipeline; user triggers it on demand via
    #         POST /api/v1/summarize/{video} or the Summary tab in the UI.
    # Recommended: false for live mode (saves 60-120s of Ollama time per window,
    # keeps the Ollama semaphore free for Ask queries sooner).
    # Batch mode can keep true since it runs offline and time is not critical.
    auto_summarize: bool = Field(True, alias="AUTO_SUMMARIZE")

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

    # ── Phase 6A: Track quality filters ──────────────────────────────────────

    # Minimum number of sampled frames a track must appear in to be kept.
    # Filters out ghost tracks (reflections, partial occlusion, 1-frame blips).
    # At 1 FPS: min_visible_frames=2 means object must appear in ≥2 seconds.
    min_visible_frames: int = Field(2, alias="MIN_VISIBLE_FRAMES")

    # Maximum time gap (seconds) between two tracks of the same class for them
    # to be considered the SAME physical object and merged.
    # Solves: security guard who sleeps/moves → ByteTrack assigns new track_id
    # on return. If gap < this value, both tracks are merged into one.
    # At 1 FPS on a 10-min video: 30s is conservative, 60s covers most cases.
    merge_gap_seconds: float = Field(30.0, alias="MERGE_GAP_SECONDS")

    # ── Phase 6B: Attribute extraction control ────────────────────────────────

    # Set false to skip all automatic Phase 6B crop analysis after YOLO.
    # 6B can always be triggered manually via POST /extract-attributes/{video}
    enable_phase_6b: bool = Field(True, alias="ENABLE_PHASE_6B")
    attribute_policy: str = Field("selective", alias="ATTRIBUTE_POLICY")
    enable_activity_captions: bool = Field(False, alias="ENABLE_ACTIVITY_CAPTIONS")
    enable_plate_ocr: bool = Field(False, alias="ENABLE_PLATE_OCR")
    attribute_min_duration_seconds: float = Field(8.0, alias="ATTRIBUTE_MIN_DURATION_SECONDS")
    attribute_max_tracks_per_window: int = Field(4, alias="ATTRIBUTE_MAX_TRACKS_PER_WINDOW")

    # Ollama workload lane (single shared queue across Ask + pipeline).
    ollama_concurrency: int = Field(1, alias="OLLAMA_CONCURRENCY")
    # Backward compatibility: older deployments still export split lane vars.
    # They are ignored for lane separation now, but accepted to avoid startup failure.
    ask_ollama_concurrency: int = Field(1, alias="ASK_OLLAMA_CONCURRENCY")
    pipeline_ollama_concurrency: int = Field(1, alias="PIPELINE_OLLAMA_CONCURRENCY")

    # ── Paths ─────────────────────────────────────────────────────────────────

    video_input_path: str = Field(..., alias="VIDEO_INPUT_PATH")
    camera_id: str = Field(..., alias="CAMERA_ID")

    # ── Live window settings ───────────────────────────────────────────────────
    live_window_minutes: int = Field(5, alias="LIVE_WINDOW_MINUTES")

    # ── Live RTSP stream settings ─────────────────────────────────────────────

    # Source URL — set VIDEO_INPUT_PATH=rtsp://... in .env for live mode

    # Frames per second to sample from live stream for detection (1-5 recommended)
    live_sample_fps: int = Field(2, alias="LIVE_SAMPLE_FPS")

    # Seconds of inactivity before a person is considered to have exited
    live_exit_timeout_seconds: int = Field(8, alias="LIVE_EXIT_TIMEOUT_SECONDS")

    # Cosine similarity threshold for matching a new detection to a known person
    # 0.75 = same clothing, reliably same person in same scene
    live_reid_threshold: float = Field(0.72, alias="LIVE_REID_THRESHOLD")


    # Max JPEG quality for MJPEG stream (lower = less bandwidth)
    live_stream_jpeg_quality: int = Field(75, alias="LIVE_STREAM_JPEG_QUALITY")

    @property
    def database_url(self) -> str:
        return (
            f"postgresql+psycopg2://{self.db_user}:"
            f"{self.db_password}@{self.db_host}:"
            f"{self.db_port}/{self.db_name}"
        )

    @property
    def is_rtsp(self) -> bool:
        """True when VIDEO_INPUT_PATH is an RTSP stream URL."""
        return self.video_input_path.lower().startswith("rtsp://")

    @property
    def live_crops_path(self) -> str:
        """Crops directory — matches Docker volume mount live_crops:/data/live_crops"""
        return "/data/live_crops"

    class Config:
        env_file = ".env"
        case_sensitive = True


@lru_cache
def get_settings() -> Settings:
    return Settings()