"""
temporal_retriever.py — Temporal-aware retrieval for video RAG.

Addresses the core feedback: RAG was purely event-based (top-k cosine hits).
Real video queries require temporal windows, track-grouped context, and
sequence reasoning ("did X happen AFTER Y?").

Three retrieval strategies, selected by query intent:

  1. TEMPORAL SEQUENCE  ("after", "before", "then", "followed by", "while")
     → parse_temporal_intent() extracts subject + event pairs
     → fetch all tracks matching subject
     → for each track, build a time-ordered window of ALL its events
     → return structured track windows for the QA engine

  2. TRACK-GROUPED     (default for behavioural/attribute queries)
     → semantic search → expand each hit to full track context
     → groups all events by track_id so LLM sees complete track history
     → adds memory graph nodes for those specific tracks

  3. MEMORY GRAPH FIRST (identity/plate/behaviour label queries)
     → search SemanticMemoryGraph.semantic_text directly via substring + embedding
     → returns matching nodes + their track context
     → most efficient for "who was loitering?", "what plate was seen?"

All strategies return a TrackContext list — a unified structure the QA engine
can render as focused, track-grouped context rather than a flat event dump.
"""

import re
from dataclasses import dataclass, field
from typing import Optional
from sqlalchemy.orm import Session
from sqlalchemy import select, asc

from app.storage.models import TrackEvent, TrackEventEmbedding, SemanticMemoryGraph, VideoTimeline
from app.rag.embedder import OllamaEmbedder
from app.core.logging import get_logger


# ── Temporal intent keywords ──────────────────────────────────────────────────

_TEMPORAL_KEYWORDS = {
    "after", "before", "then", "followed by", "while", "during",
    "when", "once", "until", "since", "later", "earlier",
    "next", "prior", "subsequently", "preceding",
}

_SEQUENCE_PAIRS = {
    # Maps common query phrases to (first_event, second_event) label pairs
    ("walk", "fall"):    ("walking", "fall_detected"),
    ("run",  "fall"):    ("running",  "fall_detected"),
    ("walk", "stop"):    ("walking",  "sudden_stop"),
    ("run",  "stop"):    ("running",  "sudden_stop"),
    ("enter","loiter"):  ("enters",   "loitering"),
    ("walk", "run"):     ("walking",  "running"),
    ("enter","exit"):    ("enters",   "exits"),
    ("enter","dwell"):   ("enters",   "dwell"),
}


@dataclass
class TrackContext:
    """
    A complete temporal context block for one physical track.
    The QA engine renders these instead of flat event lists.
    """
    video_filename: str
    track_id: int
    object_class: str
    first_seen: float
    last_seen: float
    duration: float
    best_crop_path: Optional[str]
    best_confidence: float

    # Ordered list of timeline entries for this track from VideoTimeline
    timeline_events: list = field(default_factory=list)

    # Memory graph nodes for this track
    memory_nodes: list = field(default_factory=list)

    # Raw attribute string (clothing, color, plate etc.)
    attribute_summary: str = ""

    # How this track was found
    retrieval_reason: str = ""

    # Score from embedding search (0-1)
    relevance_score: float = 0.0

    def to_context_block(self) -> str:
        """
        Render as a focused context block for the LLM prompt.
        Replaces the flat _ev_line() format with track-grouped temporal context.
        """
        lines = []
        _fmt = lambda s: f"{int(s)//60}:{int(s)%60:02d}"

        header = (
            f"── TRACK #{self.track_id} ({self.object_class.upper()}) "
            f"{_fmt(self.first_seen)}–{_fmt(self.last_seen)} "
            f"({self.duration:.0f}s) [{self.video_filename}]"
        )
        if self.attribute_summary:
            header += f" | {self.attribute_summary}"
        if self.retrieval_reason:
            header += f" | retrieved: {self.retrieval_reason}"
        lines.append(header)

        # Memory graph nodes first — most semantic
        for node in self.memory_nodes:
            lines.append(f"  [MEM:{node['node_type'].upper()}] {node['semantic_text']}")

        # Timeline events in order
        if self.timeline_events:
            lines.append("  Timeline:")
            for e in self.timeline_events:
                track_mark = ""
                lines.append(
                    f"    {e.get('time_label','?')}  "
                    f"{e.get('event',''):<20} — {e.get('detail','')}{track_mark}"
                )
        return "\n".join(lines)


def _fmt_sec(s: float) -> str:
    return f"{int(s)//60}:{int(s)%60:02d}"


def _attr_summary(attrs: dict) -> str:
    if not attrs:
        return ""
    cls = attrs.get("object_class", "")
    parts = []
    if cls in ("car", "truck", "bus", "motorcycle", "bicycle"):
        for k in ("color", "type", "make_estimate"):
            v = attrs.get(k, "")
            if v and v not in ("unknown", "none", ""):
                parts.append(v)
        plate = attrs.get("plate_number", "")
        if plate and plate not in ("unknown", ""):
            parts.append(f"plate:{plate}")
    else:
        for k in ("gender_estimate", "clothing_top", "visible_text"):
            v = attrs.get(k, "")
            if v and v not in ("unknown", "none", ""):
                parts.append(v)
        beh = (attrs.get("temporal") or {}).get("behaviour", "")
        if beh and beh not in ("unknown", "passing_through", ""):
            parts.append(f"behaviour:{beh}")
        mevts = (attrs.get("motion_summary") or {}).get("motion_events", [])
        if mevts:
            parts.append(f"motion_events:[{'; '.join(mevts[:2])}]")
    return ", ".join(parts)


class TemporalRetriever:
    """
    Temporal-aware retrieval layer on top of the existing vector search.

    Usage in QAEngine:
        tr = TemporalRetriever(db)
        intent = tr.parse_temporal_intent(question)
        if intent["is_temporal"]:
            contexts = tr.retrieve_temporal(question, video_filename)
        else:
            contexts = tr.retrieve_track_grouped(question, video_filename)
    """

    def __init__(self, db: Session, top_k: int = 8):
        self.db = db
        self.embedder = OllamaEmbedder()
        self.logger = get_logger()
        self.top_k = top_k

    # ── Intent detection ──────────────────────────────────────────────────────

    def parse_temporal_intent(self, query: str) -> dict:
        """
        Detect whether a query requires temporal/sequence reasoning.

        Returns:
          {
            "is_temporal": bool,
            "keywords_found": list[str],
            "sequence_hint": (first_label, second_label) | None,
            "subjects": list[str],   # object classes mentioned
          }
        """
        q_lower = query.lower()
        words = set(re.findall(r"\b\w+\b", q_lower))

        # Check for temporal keywords
        found_keywords = [kw for kw in _TEMPORAL_KEYWORDS if kw in q_lower]

        # Check for known sequence pairs
        sequence_hint = None
        for (w1, w2), (e1, e2) in _SEQUENCE_PAIRS.items():
            if w1 in q_lower and w2 in q_lower:
                sequence_hint = (e1, e2)
                break

        # Detect object class mentions
        subjects = []
        for cls in ("person", "people", "car", "vehicle", "truck", "bus"):
            if cls in q_lower:
                subjects.append("person" if cls == "people" else cls)

        is_temporal = bool(found_keywords) or sequence_hint is not None

        return {
            "is_temporal": is_temporal,
            "keywords_found": found_keywords,
            "sequence_hint": sequence_hint,
            "subjects": list(set(subjects)),
        }

    # ── Strategy 1: Temporal sequence retrieval ───────────────────────────────

    def retrieve_temporal(
        self,
        query: str,
        video_filename: Optional[str] = None,
        intent: Optional[dict] = None,
    ) -> list[TrackContext]:
        """
        For temporal queries: find relevant tracks via semantic search,
        then build a COMPLETE temporal window for each track showing
        all its events in order.

        This is what answers "did person fall AFTER walking?" — both
        the walking and fall events appear in the same TrackContext block.
        """
        if intent is None:
            intent = self.parse_temporal_intent(query)

        # Step 1: semantic search to find candidate track IDs
        query_vector = self.embedder.embed(query)
        candidate_track_ids = self._semantic_track_search(query_vector, video_filename)

        # Step 2: for each candidate track, build full temporal context
        contexts = []
        seen_tracks = set()

        for video_fn, track_id in candidate_track_ids:
            key = (video_fn, track_id)
            if key in seen_tracks:
                continue
            seen_tracks.add(key)

            ctx = self._build_track_context(
                video_fn, track_id,
                retrieval_reason="temporal_sequence"
            )
            if ctx:
                contexts.append(ctx)

        # Step 3: if sequence_hint exists, filter to tracks that actually
        # contain both events of the sequence
        if intent.get("sequence_hint"):
            e1, e2 = intent["sequence_hint"]
            filtered = []
            for ctx in contexts:
                event_labels = {e.get("event", "") for e in ctx.timeline_events}
                # Include if track has either event (sequence may be partial)
                if e1 in event_labels or e2 in event_labels:
                    filtered.append(ctx)
            # Fall back to all if filter removed everything
            if filtered:
                contexts = filtered

        # Step 4: if we still have nothing, try memory graph search
        if not contexts:
            contexts = self._memory_graph_search(query, video_filename)

        return contexts[:self.top_k]

    # ── Strategy 2: Track-grouped retrieval (default) ─────────────────────────

    def retrieve_track_grouped(
        self,
        query: str,
        video_filename: Optional[str] = None,
    ) -> list[TrackContext]:
        """
        Default retrieval: semantic search → expand each hit to full track.
        Groups all events by track_id so LLM sees complete track history.
        """
        query_vector = self.embedder.embed(query)
        candidate_track_ids = self._semantic_track_search(query_vector, video_filename)

        contexts = []
        seen = set()
        for video_fn, track_id in candidate_track_ids:
            key = (video_fn, track_id)
            if key in seen:
                continue
            seen.add(key)
            ctx = self._build_track_context(
                video_fn, track_id,
                retrieval_reason="semantic_match"
            )
            if ctx:
                contexts.append(ctx)

        # Also search memory graph directly for additional relevant tracks
        mem_contexts = self._memory_graph_search(query, video_filename)
        for ctx in mem_contexts:
            key = (ctx.video_filename, ctx.track_id)
            if key not in seen and ctx.track_id is not None:
                seen.add(key)
                contexts.append(ctx)

        return contexts[:self.top_k]

    # ── Strategy 3: Memory graph first ───────────────────────────────────────

    def _memory_graph_search(
        self,
        query: str,
        video_filename: Optional[str] = None,
    ) -> list[TrackContext]:
        """
        Search SemanticMemoryGraph nodes by substring match on semantic_text.
        Much more effective for behaviour/identity queries than embedding search
        because memory nodes contain pre-digested natural language facts.
        """
        q_lower = query.lower()
        # Extract meaningful words (ignore stop words)
        stop = {"did", "the", "a", "an", "is", "was", "were", "has", "have",
                "any", "anyone", "in", "at", "on", "of", "to", "do", "does"}
        query_words = [w for w in re.findall(r"\b\w+\b", q_lower) if w not in stop and len(w) > 2]

        db_q = self.db.query(SemanticMemoryGraph)
        if video_filename:
            db_q = db_q.filter(SemanticMemoryGraph.video_filename == video_filename)
        nodes = db_q.all()

        scored_nodes = []
        for node in nodes:
            text_lower = node.semantic_text.lower()
            label_lower = node.node_label.lower()
            matched = sum(1 for w in query_words if w in text_lower or w in label_lower)
            if matched > 0:
                score = matched / max(len(query_words), 1)
                scored_nodes.append((score, node))

        scored_nodes.sort(key=lambda x: x[0], reverse=True)

        # Build TrackContext for each matched node's track
        seen = set()
        contexts = []
        for score, node in scored_nodes[:self.top_k]:
            if node.track_id is None:
                continue
            key = (node.video_filename, node.track_id)
            if key in seen:
                continue
            seen.add(key)
            ctx = self._build_track_context(
                node.video_filename, node.track_id,
                retrieval_reason=f"memory_graph:{node.node_type}",
                base_score=score,
            )
            if ctx:
                contexts.append(ctx)

        return contexts

    # ── Core: build a complete TrackContext for one track ─────────────────────

    def _build_track_context(
        self,
        video_filename: str,
        track_id: int,
        retrieval_reason: str = "",
        base_score: float = 0.0,
    ) -> Optional[TrackContext]:
        """
        Build a complete TrackContext for a single track_id:
        - Entry event (attributes, crop, timing)
        - All timeline events for this track from VideoTimeline
        - All memory graph nodes for this track
        """
        # Get the entry event (canonical record for this track)
        entry = (
            self.db.query(TrackEvent)
            .filter(
                TrackEvent.video_filename == video_filename,
                TrackEvent.track_id == track_id,
                TrackEvent.event_type == "entry",
            )
            .first()
        )
        if not entry:
            return None

        attrs = entry.attributes or {}

        # Get timeline events for this track from VideoTimeline
        tl = (
            self.db.query(VideoTimeline)
            .filter(VideoTimeline.video_filename == video_filename)
            .first()
        )
        track_timeline_events = []
        if tl and tl.timeline_entries:
            track_timeline_events = [
                e for e in tl.timeline_entries
                if e.get("track_id") == track_id
            ]
            # Sort by second
            track_timeline_events.sort(key=lambda e: e.get("second", 0))

        # Get memory graph nodes for this track
        mem_nodes = (
            self.db.query(SemanticMemoryGraph)
            .filter(
                SemanticMemoryGraph.video_filename == video_filename,
                SemanticMemoryGraph.track_id == track_id,
            )
            .order_by(SemanticMemoryGraph.node_type)
            .all()
        )

        return TrackContext(
            video_filename=video_filename,
            track_id=track_id,
            object_class=entry.object_class,
            first_seen=entry.first_seen_second,
            last_seen=entry.last_seen_second,
            duration=entry.duration_seconds,
            best_crop_path=entry.best_crop_path,
            best_confidence=entry.best_confidence or 0.0,
            timeline_events=track_timeline_events,
            memory_nodes=[
                {
                    "node_type": n.node_type,
                    "node_label": n.node_label,
                    "semantic_text": n.semantic_text,
                }
                for n in mem_nodes
            ],
            attribute_summary=_attr_summary(attrs),
            retrieval_reason=retrieval_reason,
            relevance_score=base_score,
        )

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _semantic_track_search(
        self,
        query_vector,
        video_filename: Optional[str] = None,
    ) -> list[tuple[str, int]]:
        """
        Semantic search over TrackEventEmbedding.
        Returns list of (video_filename, track_id) pairs, deduplicated.
        """
        stmt = (
            select(
                TrackEvent.video_filename,
                TrackEvent.track_id,
                TrackEventEmbedding.embedding.cosine_distance(query_vector).label("distance"),
            )
            .join(TrackEventEmbedding, TrackEventEmbedding.track_event_id == TrackEvent.id)
        )
        if video_filename:
            stmt = stmt.where(TrackEvent.video_filename == video_filename)

        stmt = stmt.order_by("distance").limit(self.top_k * 3)  # over-fetch then dedup
        rows = self.db.execute(stmt).fetchall()

        seen = set()
        result = []
        for row in rows:
            key = (row.video_filename, row.track_id)
            if key not in seen:
                seen.add(key)
                result.append(key)

        return result[:self.top_k]

    def render_contexts_as_text(self, contexts: list[TrackContext]) -> str:
        """
        Render a list of TrackContext objects as the structured context
        block that goes into the QA prompt.
        """
        if not contexts:
            return ""
        lines = ["=== TEMPORAL TRACK CONTEXT (grouped by physical object) ==="]
        for ctx in contexts:
            lines.append("")
            lines.append(ctx.to_context_block())
        return "\n".join(lines)