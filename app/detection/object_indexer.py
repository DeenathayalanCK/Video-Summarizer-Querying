from sqlalchemy.orm import Session

from app.storage.models import DetectedObject, DetectedObjectEmbedding, TrackEvent, TrackEventEmbedding
from app.rag.embedder import OllamaEmbedder
from app.core.config import get_settings
from app.core.logging import get_logger


class ObjectIndexer:
    """
    Embeds DetectedObject and TrackEvent rag_text into pgvector.
    Parallel to the existing CaptionIndexer — same pattern, new tables.
    """

    def __init__(self, db: Session):
        self.db = db
        self.embedder = OllamaEmbedder()
        self.settings = get_settings()
        self.logger = get_logger()

    def index_detected_object(self, obj: DetectedObject) -> DetectedObjectEmbedding:
        """Embed a single DetectedObject's rag_text."""
        existing = (
            self.db.query(DetectedObjectEmbedding)
            .filter(DetectedObjectEmbedding.object_id == obj.id)
            .first()
        )
        if existing:
            return existing

        if not obj.rag_text:
            self.logger.warning("detected_object_missing_rag_text", object_id=str(obj.id))
            return None

        vector = self.embedder.embed(obj.rag_text)
        emb = DetectedObjectEmbedding(
            object_id=obj.id,
            embedding=vector,
            model_name=self.settings.embed_model,
        )
        self.db.add(emb)
        self.db.commit()
        self.db.refresh(emb)
        return emb

    def index_track_event(self, event: TrackEvent) -> TrackEventEmbedding:
        """Embed a single TrackEvent's rag_text."""
        existing = (
            self.db.query(TrackEventEmbedding)
            .filter(TrackEventEmbedding.track_event_id == event.id)
            .first()
        )
        if existing:
            return existing

        if not event.rag_text:
            self.logger.warning("track_event_missing_rag_text", event_id=str(event.id))
            return None

        vector = self.embedder.embed(event.rag_text)
        emb = TrackEventEmbedding(
            track_event_id=event.id,
            embedding=vector,
            model_name=self.settings.embed_model,
        )
        self.db.add(emb)
        self.db.commit()
        self.db.refresh(emb)
        return emb

    def index_all_unindexed_objects(self) -> int:
        """Find all DetectedObjects with no embedding and index them."""
        unindexed = (
            self.db.query(DetectedObject)
            .outerjoin(
                DetectedObjectEmbedding,
                DetectedObjectEmbedding.object_id == DetectedObject.id
            )
            .filter(DetectedObjectEmbedding.id == None)
            .filter(DetectedObject.rag_text != None)
            .all()
        )
        for obj in unindexed:
            self.index_detected_object(obj)
        return len(unindexed)

    def index_all_unindexed_events(self) -> int:
        """Find all TrackEvents with no embedding and index them."""
        unindexed = (
            self.db.query(TrackEvent)
            .outerjoin(
                TrackEventEmbedding,
                TrackEventEmbedding.track_event_id == TrackEvent.id
            )
            .filter(TrackEventEmbedding.id == None)
            .filter(TrackEvent.rag_text != None)
            .all()
        )
        for event in unindexed:
            self.index_track_event(event)
        return len(unindexed)