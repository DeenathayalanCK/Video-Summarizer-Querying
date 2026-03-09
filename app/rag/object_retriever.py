from sqlalchemy.orm import Session
from sqlalchemy import select
from typing import Optional

from app.storage.models import (
    DetectedObject, DetectedObjectEmbedding,
    TrackEvent, TrackEventEmbedding,
)
from app.rag.embedder import OllamaEmbedder
from app.core.logging import get_logger


class ObjectRetriever:
    """
    Semantic search over Phase 6A detection data.

    Two search targets:
    1. DetectedObject — individual frame-level detections
       Good for: "show me every frame with a person"
    2. TrackEvent — lifecycle events (entry/exit/dwell)
       Good for: "did any vehicle enter?", "was there loitering?"

    Both use pgvector cosine similarity over their rag_text embeddings.
    """

    def __init__(self, db: Session, top_k: int = 8):
        self.db = db
        self.top_k = top_k
        self.embedder = OllamaEmbedder()
        self.logger = get_logger()

    def search_detections(
        self,
        query: str,
        video_filename: Optional[str] = None,
        camera_id: Optional[str] = None,
        object_class: Optional[str] = None,
        min_second: Optional[float] = None,
        max_second: Optional[float] = None,
    ) -> list[dict]:
        """
        Search DetectedObject table by semantic similarity.
        Returns individual frame-level detections ordered by relevance.
        """
        self.logger.info("object_retriever_searching_detections", query=query)
        query_vector = self.embedder.embed(query)

        stmt = (
            select(
                DetectedObject.id,
                DetectedObject.video_filename,
                DetectedObject.camera_id,
                DetectedObject.frame_second_offset,
                DetectedObject.object_class,
                DetectedObject.confidence,
                DetectedObject.track_id,
                DetectedObject.frame_quadrant,
                DetectedObject.crop_path,
                DetectedObject.bbox_x1,
                DetectedObject.bbox_y1,
                DetectedObject.bbox_x2,
                DetectedObject.bbox_y2,
                DetectedObject.rag_text,
                DetectedObjectEmbedding.embedding.cosine_distance(query_vector).label("distance"),
            )
            .join(DetectedObjectEmbedding, DetectedObjectEmbedding.object_id == DetectedObject.id)
        )

        if video_filename:
            stmt = stmt.where(DetectedObject.video_filename == video_filename)
        if camera_id:
            stmt = stmt.where(DetectedObject.camera_id == camera_id)
        if object_class:
            stmt = stmt.where(DetectedObject.object_class == object_class)
        if min_second is not None:
            stmt = stmt.where(DetectedObject.frame_second_offset >= min_second)
        if max_second is not None:
            stmt = stmt.where(DetectedObject.frame_second_offset <= max_second)

        stmt = stmt.order_by("distance").limit(self.top_k)
        rows = self.db.execute(stmt).fetchall()

        return [
            {
                "object_id": str(row.id),
                "video_filename": row.video_filename,
                "camera_id": row.camera_id,
                "second": row.frame_second_offset,
                "object_class": row.object_class,
                "confidence": round(row.confidence, 3),
                "track_id": row.track_id,
                "quadrant": row.frame_quadrant,
                "crop_path": row.crop_path,
                "bbox": {
                    "x1": row.bbox_x1, "y1": row.bbox_y1,
                    "x2": row.bbox_x2, "y2": row.bbox_y2,
                },
                "rag_text": row.rag_text,
                "score": round(1 - row.distance, 4),
            }
            for row in rows
        ]

    def search_track_events(
        self,
        query: str,
        video_filename: Optional[str] = None,
        camera_id: Optional[str] = None,
        event_type: Optional[str] = None,
        object_class: Optional[str] = None,
    ) -> list[dict]:
        """
        Search TrackEvent table by semantic similarity.
        Returns lifecycle events ordered by relevance.
        """
        self.logger.info("object_retriever_searching_events", query=query)
        query_vector = self.embedder.embed(query)

        stmt = (
            select(
                TrackEvent.id,
                TrackEvent.video_filename,
                TrackEvent.camera_id,
                TrackEvent.track_id,
                TrackEvent.object_class,
                TrackEvent.event_type,
                TrackEvent.first_seen_second,
                TrackEvent.last_seen_second,
                TrackEvent.duration_seconds,
                TrackEvent.best_frame_second,
                TrackEvent.best_crop_path,
                TrackEvent.best_confidence,
                TrackEvent.rag_text,
                TrackEvent.attributes,
                TrackEventEmbedding.embedding.cosine_distance(query_vector).label("distance"),
            )
            .join(TrackEventEmbedding, TrackEventEmbedding.track_event_id == TrackEvent.id)
        )

        if video_filename:
            stmt = stmt.where(TrackEvent.video_filename == video_filename)
        if camera_id:
            stmt = stmt.where(TrackEvent.camera_id == camera_id)
        if event_type:
            stmt = stmt.where(TrackEvent.event_type == event_type)
        if object_class:
            stmt = stmt.where(TrackEvent.object_class == object_class)

        stmt = stmt.order_by("distance").limit(self.top_k)
        rows = self.db.execute(stmt).fetchall()
        results = [
            {
                "event_id": str(row.id),
                "video_filename": row.video_filename,
                "camera_id": row.camera_id,
                "track_id": row.track_id,
                "object_class": row.object_class,
                "event_type": row.event_type,
                "first_seen": row.first_seen_second,
                "last_seen": row.last_seen_second,
                "duration": row.duration_seconds,
                "best_frame_second": row.best_frame_second,
                "best_crop_path": row.best_crop_path,
                "best_confidence": round(row.best_confidence or 0, 3),
                "rag_text": row.rag_text,
                "attributes": row.attributes,
                "score": round(1 - row.distance, 4),
            }
            for row in rows
        ]

        # Fallback: if embedding search returned nothing, return recent entry events.
        # Covers live windows before _run_reembed runs.
        if not results:
            try:
                q = self.db.query(TrackEvent).filter(TrackEvent.event_type == "entry")
                if video_filename:
                    q = q.filter(TrackEvent.video_filename == video_filename)
                if camera_id:
                    q = q.filter(TrackEvent.camera_id == camera_id)
                if object_class:
                    q = q.filter(TrackEvent.object_class == object_class)
                evs = q.order_by(TrackEvent.first_seen_second.desc()).limit(self.top_k).all()
                for ev in evs:
                    results.append({
                        "event_id": str(ev.id),
                        "video_filename": ev.video_filename,
                        "camera_id": ev.camera_id,
                        "track_id": ev.track_id,
                        "object_class": ev.object_class,
                        "event_type": ev.event_type,
                        "first_seen": ev.first_seen_second,
                        "last_seen": ev.last_seen_second,
                        "duration": ev.duration_seconds,
                        "best_frame_second": ev.best_frame_second,
                        "best_crop_path": ev.best_crop_path,
                        "best_confidence": round(ev.best_confidence or 0, 3),
                        "rag_text": ev.rag_text,
                        "attributes": ev.attributes,
                        "score": 0.5,  # neutral score for fallback results
                    })
            except Exception as e:
                self.logger.debug("search_track_events_fallback_failed", error=str(e))

        return results

    def attribute_keyword_search(
        self,
        query: str,
        video_filename=None,
        event_type=None,
        object_class=None,
        top_k: int = 8,
    ) -> list[dict]:
        """
        Keyword search directly against TrackEvent.attributes JSONB.
        Complements semantic search — catches color/type/clothing queries
        that semantic search misses because the query is too short or specific.

        Works by fetching all track events with attributes and doing
        Python-side substring matching. Fast enough for small datasets.
        """
        from app.storage.models import TrackEvent

        q_lower = query.lower()
        query_words = set(q_lower.split())

        q = (
            self.db.query(TrackEvent)
            .filter(TrackEvent.attributes.isnot(None))
        )
        if video_filename:
            q = q.filter(TrackEvent.video_filename == video_filename)
        if event_type:
            q = q.filter(TrackEvent.event_type == event_type)
        if object_class:
            q = q.filter(TrackEvent.object_class == object_class)

        events = q.all()
        results = []

        for ev in events:
            attrs = ev.attributes or {}
            # Build a flat string of all attribute values for matching
            attr_values = " ".join(str(v).lower() for v in attrs.values() if v)
            rag_lower = (ev.rag_text or "").lower()
            combined = attr_values + " " + rag_lower

            # Score = fraction of query words found in attributes
            matched = sum(1 for w in query_words if w in combined)
            if matched == 0:
                continue

            score = matched / len(query_words)
            # Boost if match is in attribute values specifically (not just rag_text)
            attr_matched = sum(1 for w in query_words if w in attr_values)
            if attr_matched > 0:
                score = min(1.0, score + 0.3)

            results.append({
                "event_id": str(ev.id),
                "video_filename": ev.video_filename,
                "camera_id": ev.camera_id,
                "track_id": ev.track_id,
                "object_class": ev.object_class,
                "event_type": ev.event_type,
                "first_seen": ev.first_seen_second,
                "last_seen": ev.last_seen_second,
                "duration": ev.duration_seconds,
                "best_frame_second": ev.best_frame_second,
                "best_crop_path": ev.best_crop_path,
                "best_confidence": round(ev.best_confidence or 0, 3),
                "rag_text": ev.rag_text,
                "attributes": ev.attributes,
                "score": round(score, 4),
                "match_type": "attribute_keyword",
            })

        # Sort by score descending, deduplicate by event_id
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k]

    def get_track_timeline(self, video_filename: str) -> list[dict]:
        """
        Return one timeline entry per unique tracked object, in chronological order.

        We use only ENTRY events — they already carry first_seen, last_seen,
        and duration. Showing entry + exit + dwell for the same track produces
        visual duplicates in the UI (same timestamp, same crop, same text).

        Each card in the timeline represents one physical object's full lifespan.
        """
        rows = (
            self.db.query(TrackEvent)
            .filter(
                TrackEvent.video_filename == video_filename,
                TrackEvent.event_type == "entry",   # one row per track
            )
            .order_by(TrackEvent.first_seen_second)
            .all()
        )
        return [
            {
                "track_id": r.track_id,
                "object_class": r.object_class,
                "event_type": r.event_type,
                "first_seen": r.first_seen_second,
                "last_seen": r.last_seen_second,
                "duration": r.duration_seconds,
                "best_frame_second": r.best_frame_second,
                "best_crop_path": r.best_crop_path,
                "best_confidence": round(r.best_confidence or 0, 3),
                "attributes": r.attributes,
                # Convenience flags for timeline card rendering
                "has_dwell": r.duration_seconds >= 10.0,
            }
            for r in rows
        ]