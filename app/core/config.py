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

    # Vision model — used for frame captioning (must be multimodal)
    multimodal_model: str = Field(..., alias="MULTIMODAL_MODEL")

    # Text model — used for Q&A and summarization (text-only LLM)
    text_model: str = Field(..., alias="TEXT_MODEL")

    # Embedding model — used for semantic search indexing
    embed_model: str = Field(..., alias="EMBED_MODEL")

    frame_sample_fps: int = Field(..., alias="FRAME_SAMPLE_FPS")
    scene_change_threshold: float = Field(..., alias="SCENE_CHANGE_THRESHOLD")

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