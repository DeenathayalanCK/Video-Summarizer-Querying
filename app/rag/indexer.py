from sqlalchemy.orm import Session

from app.storage.models import Caption, CaptionEmbedding
from app.rag.embedder import OllamaEmbedder
from app.core.config import get_settings
from app.core.logging import get_logger


class CaptionIndexer:
    """
    After a caption is saved, this indexes it into pgvector
    so it becomes searchable via semantic similarity.
    """

    def __init__(self, db: Session):
        self.db = db
        self.embedder = OllamaEmbedder()
        self.settings = get_settings()
        self.logger = get_logger()

    def index_caption(self, caption: Caption) -> CaptionEmbedding:
        """Embed a single caption and store the vector."""

        existing = (
            self.db.query(CaptionEmbedding)
            .filter(CaptionEmbedding.caption_id == caption.id)
            .first()
        )
        if existing:
            self.logger.info("caption_already_indexed", caption_id=str(caption.id))
            return existing

        vector = self.embedder.embed(caption.caption_text)

        emb = CaptionEmbedding(
            caption_id=caption.id,
            embedding=vector,
            model_name=self.settings.embed_model,
        )

        self.db.add(emb)
        self.db.commit()
        self.db.refresh(emb)

        self.logger.info(
            "caption_indexed",
            caption_id=str(caption.id),
            video=caption.video_filename,
            second=caption.frame_second_offset,
        )

        return emb

    def index_all_unindexed(self) -> int:
        """Find all captions with no embedding and index them."""
        unindexed = (
            self.db.query(Caption)
            .outerjoin(CaptionEmbedding, CaptionEmbedding.caption_id == Caption.id)
            .filter(CaptionEmbedding.id == None)
            .all()
        )

        self.logger.info("indexing_unindexed_captions", count=len(unindexed))

        for caption in unindexed:
            self.index_caption(caption)

        return len(unindexed)