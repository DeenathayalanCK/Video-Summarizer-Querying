from sqlalchemy.orm import Session
from sqlalchemy import select, text
from app.storage.models import Caption, CaptionEmbedding
from app.rag.embedder import OllamaEmbedder
from app.core.logging import get_logger


class CaptionRetriever:
    """
    Given a natural language query, returns the top-K most semantically
    relevant captions using cosine similarity over pgvector embeddings.
    """

    def __init__(self, db: Session, top_k: int = 8):
        self.db = db
        self.top_k = top_k
        self.embedder = OllamaEmbedder()
        self.logger = get_logger()

    def search(
        self,
        query: str,
        video_filename: str = None,
        camera_id: str = None,
        min_second: float = None,
        max_second: float = None,
    ) -> list[dict]:
        """
        Embed the query and find the closest captions by cosine similarity.
        Optionally filter by video, camera, or time range.
        Returns a list of result dicts ordered by relevance.
        """
        self.logger.info("retriever_searching", query=query)

        query_vector = self.embedder.embed(query)

        # Build base query joining captions + embeddings
        stmt = (
            select(
                Caption.id,
                Caption.video_filename,
                Caption.camera_id,
                Caption.frame_second_offset,
                Caption.caption_text,
                Caption.keyframe_path,
                Caption.created_at,
                CaptionEmbedding.embedding.cosine_distance(query_vector).label("distance"),
            )
            .join(CaptionEmbedding, CaptionEmbedding.caption_id == Caption.id)
        )

        # Optional filters
        if video_filename:
            stmt = stmt.where(Caption.video_filename == video_filename)
        if camera_id:
            stmt = stmt.where(Caption.camera_id == camera_id)
        if min_second is not None:
            stmt = stmt.where(Caption.frame_second_offset >= min_second)
        if max_second is not None:
            stmt = stmt.where(Caption.frame_second_offset <= max_second)

        stmt = stmt.order_by("distance").limit(self.top_k)

        rows = self.db.execute(stmt).fetchall()

        results = [
            {
                "caption_id": str(row.id),
                "video_filename": row.video_filename,
                "camera_id": row.camera_id,
                "second": row.frame_second_offset,
                "caption": row.caption_text,
                "keyframe_path": row.keyframe_path,
                "score": round(1 - row.distance, 4),  # convert distance → similarity
                "created_at": row.created_at.isoformat() if row.created_at else None,
            }
            for row in rows
        ]

        self.logger.info("retriever_results", count=len(results))
        return results

    def get_timeline(self, video_filename: str) -> list[dict]:
        """Return all captions for a video in chronological order."""
        rows = (
            self.db.query(Caption)
            .filter(Caption.video_filename == video_filename)
            .order_by(Caption.frame_second_offset)
            .all()
        )

        return [
            {
                "caption_id": str(r.id),
                "second": r.frame_second_offset,
                "caption": r.caption_text,
                "keyframe_path": r.keyframe_path,
            }
            for r in rows
        ]