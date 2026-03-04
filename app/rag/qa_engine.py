"""
qa_engine.py — Semantic Q&A over video surveillance data.

Upgraded context builder feeds the LLM FOUR layers of information:

  Layer 1 — Video Summary       : narrative text from VideoSummarizer
  Layer 2 — Semantic Memory     : graph nodes from SemanticMemoryGraph
  Layer 3 — Behaviour context   : temporal + motion_summary from TrackEvent.attributes
  Layer 4 — Raw detection events: chronological ENTRY/EXIT/DWELL rows (with attr strings)

The LLM previously only saw Layer 1 + Layer 4.
Now it sees all four, so queries like:
  "was anyone loitering?"       → Layer 3 has behaviour="loitering"
  "did anyone fall?"            → Layer 3 has motion_events=["fall_proxy @ 32s"]
  "what was the person doing?"  → Layer 3 has dominant_motion + notes
  "what happened at 0:09?"      → Layer 2 memory nodes + VideoTimeline entries
"""

import requests
from collections import defaultdict
from sqlalchemy.orm import Session
from sqlalchemy import asc

from app.core.config import get_settings
from app.core.logging import get_logger
from app.rag.retriever import CaptionRetriever
from app.rag.object_retriever import ObjectRetriever
from app.rag.summarizer import _condense_caption
from app.storage.repository import EventRepository
from app.storage.models import Caption, TrackEvent, VideoSummary
from app.prompts.qa_prompt import (
    QA_SYSTEM_PROMPT,
    QA_USER_TEMPLATE,
    QA_DETECTION_SYSTEM_PROMPT,
    QA_DETECTION_USER_TEMPLATE,
)

_QA_MAX_EVENTS = 150


def _fmt_sec(s):
    return f"{int(s) // 60}:{int(s) % 60:02d}"


def _build_attr_str(ev):
    if not ev.attributes:
        return ""
    attrs = ev.attributes
    cls = attrs.get("object_class", ev.object_class)
    if cls in ("car", "truck", "bus", "motorcycle", "bicycle"):
        color = attrs.get("color", "")
        vtype = attrs.get("type", "")
        make  = attrs.get("make_estimate", "")
        parts = [p for p in [color, vtype] if p and p != "unknown"]
        if make and make != "unknown":
            parts.append(f"({make})")
        return " ".join(parts)
    elif cls == "person":
        gender   = attrs.get("gender_estimate", "")
        age      = attrs.get("age_estimate", "")
        top      = attrs.get("clothing_top", "")
        bottom   = attrs.get("clothing_bottom", "")
        carrying = attrs.get("carrying", "")
        vis      = attrs.get("visible_text", "")
        parts = [p for p in [gender, age, top, bottom]
                 if p and p not in ("unknown", "none")]
        if carrying and carrying not in ("unknown", "none"):
            parts.append(f"carrying {carrying}")
        if vis and vis not in ("none", "unknown"):
            parts.append(f'"{vis}"')
        return ", ".join(parts)
    return ""


def _build_temporal_str(ev) -> str:
    """
    Build a semantic behaviour string from TrackEvent.attributes["temporal"]
    and attributes["motion_summary"].  Returns empty string if no data.

    Examples:
      → "behaviour=loitering, motion=walking, notes=Person present 64s..."
      → "behaviour=fall_detected, motion_events=[fall_proxy @ 32.0s]"
    """
    if not ev.attributes:
        return ""
    attrs   = ev.attributes
    temp    = attrs.get("temporal", {})
    ms      = attrs.get("motion_summary", {})

    parts = []

    if temp:
        b = temp.get("behaviour", "")
        if b and b != "unknown":
            parts.append(f"behaviour={b}")
        dom = temp.get("dominant_motion", "") or ms.get("dominant_state", "")
        if dom and dom not in ("unknown", "stationary"):
            parts.append(f"motion={dom}")
        notes = temp.get("notes", "")
        if notes:
            parts.append(f'notes="{notes}"')
        appc = temp.get("appearance_count", 1)
        if appc > 1:
            parts.append(f"appeared_times={appc}")
        mvmt = temp.get("movement_pattern", "")
        if mvmt and mvmt != "unknown":
            parts.append(f"movement={mvmt}")

    if ms:
        mevts = ms.get("motion_events", [])
        if mevts:
            parts.append(f"motion_events=[{'; '.join(mevts[:3])}]")
        direction = ms.get("direction", "")
        if direction and direction not in ("stationary", "unknown"):
            parts.append(f"direction={direction}")

    return ", ".join(parts)


def _ev_line(vf, ev) -> str:
    """Single detection event line — now includes temporal/motion semantics."""
    ad = _build_attr_str(ev)
    obj_desc = f"{ad} {ev.object_class}".strip() if ad else ev.object_class
    base = (
        f"[{vf} @ {_fmt_sec(ev.first_seen_second)}-{_fmt_sec(ev.last_seen_second)}] "
        f"{ev.event_type.upper()}: {obj_desc} track #{ev.track_id} "
        f"(duration: {ev.duration_seconds:.0f}s, conf: {ev.best_confidence or 0:.0%})"
    )
    semantic = _build_temporal_str(ev)
    return f"{base} [{semantic}]" if semantic else base


def _build_behaviour_context(db, involved_videos: set) -> str:
    """
    Layer 3: extract per-track behaviour summaries from TrackEvent.attributes.
    Groups by video → track → one line per entry event.
    Only entry events carry full attributes.
    """
    lines = ["=== BEHAVIOUR ANALYSIS (per track) ==="]
    found_any = False

    for vf in sorted(involved_videos):
        entry_evs = (
            db.query(TrackEvent)
            .filter(
                TrackEvent.video_filename == vf,
                TrackEvent.event_type == "entry",
            )
            .order_by(asc(TrackEvent.first_seen_second))
            .all()
        )
        for ev in entry_evs:
            sem = _build_temporal_str(ev)
            if not sem:
                continue
            found_any = True
            ad = _build_attr_str(ev)
            desc = f"{ad} {ev.object_class}".strip() if ad else ev.object_class
            lines.append(
                f"[{vf}] Track #{ev.track_id} ({desc}) "
                f"{_fmt_sec(ev.first_seen_second)}-{_fmt_sec(ev.last_seen_second)}: "
                f"{sem}"
            )

    if not found_any:
        return ""   # skip this section if no temporal data exists
    return "\n".join(lines)


def _build_timeline_context(db, involved_videos: set, max_entries: int = 80) -> str:
    """
    Layer 2b: pull VideoTimeline.timeline_entries for relevant videos.
    Gives the LLM the per-second event spine — much richer than raw events.
    Caps at max_entries to stay within context window.
    """
    from app.storage.models import VideoTimeline

    lines = ["=== VIDEO TIMELINE (per-second events) ==="]
    found_any = False

    for vf in sorted(involved_videos):
        tl = (
            db.query(VideoTimeline)
            .filter(VideoTimeline.video_filename == vf)
            .first()
        )
        if not tl or not tl.timeline_entries:
            continue
        found_any = True
        entries = tl.timeline_entries[:max_entries]
        lines.append(f"[{vf}]")
        for e in entries:
            track_str = f" (track #{e['track_id']})" if e.get("track_id") is not None else ""
            lines.append(
                f"  {e['time_label']}  {e['event']:<18} {e['object_class']}{track_str}"
                f"  — {e.get('detail','')}"
            )
        if len(tl.timeline_entries) > max_entries:
            lines.append(f"  ... ({len(tl.timeline_entries) - max_entries} more entries)")

        # Also include scene events
        for se in (tl.scene_events or []):
            lines.append(
                f"  SCENE_EVENT: {se['event_type']} "
                f"{_fmt_sec(se['start_second'])}-{_fmt_sec(se['end_second'])} "
                f"conf={se.get('confidence',0):.0%} — {se.get('notes','')}"
            )

    if not found_any:
        return ""
    return "\n".join(lines)


def _build_memory_context(db, involved_videos: set) -> str:
    """
    Layer 2a: pull SemanticMemoryGraph nodes for relevant videos.
    Each node is a rich semantic fact — identity, behaviour, relationships.
    """
    from app.storage.models import SemanticMemoryGraph

    lines = ["=== SEMANTIC MEMORY (knowledge graph) ==="]
    found_any = False

    for vf in sorted(involved_videos):
        graphs = (
            db.query(SemanticMemoryGraph)
            .filter(SemanticMemoryGraph.video_filename == vf)
            .order_by(SemanticMemoryGraph.node_type, SemanticMemoryGraph.track_id)
            .all()
        )
        if not graphs:
            continue
        found_any = True
        lines.append(f"[{vf}]")
        for g in graphs:
            lines.append(f"  {g.node_type.upper():12} {g.node_label:<30} {g.semantic_text}")

    if not found_any:
        return ""
    return "\n".join(lines)


def _build_raw_events_context(db, involved_videos: set, logger) -> str:
    """Layer 4: raw chronological detection events (original context builder)."""
    all_pairs = []
    for vf in sorted(involved_videos):
        evs = (
            db.query(TrackEvent)
            .filter(TrackEvent.video_filename == vf)
            .order_by(asc(TrackEvent.first_seen_second))
            .all()
        )
        all_pairs.extend((vf, ev) for ev in evs)

    if len(all_pairs) <= _QA_MAX_EVENTS:
        return "\n".join(_ev_line(vf, ev) for vf, ev in all_pairs)

    logger.info("qa_context_guard_triggered",
                total=len(all_pairs), cap=_QA_MAX_EVENTS)

    lines = [
        f"[NOTE: {len(all_pairs)} detection events total — "
        f"showing grouped summary + key events]"
    ]

    by_vc = defaultdict(list)
    for vf, ev in all_pairs:
        if ev.event_type == "entry":
            by_vc[(vf, ev.object_class)].append(ev)

    lines.append("--- Class summary ---")
    for (vf, cls), group in sorted(by_vc.items()):
        durs = [e.duration_seconds for e in group]
        lines.append(
            f"[{vf}] {cls.upper()}: {len(group)} unique tracks, "
            f"durations {min(durs):.0f}s-{max(durs):.0f}s "
            f"(avg {sum(durs) / len(durs):.0f}s)"
        )

    keep = _QA_MAX_EVENTS // 3
    lines.append("--- First events ---")
    for vf, ev in all_pairs[:keep]:
        lines.append(_ev_line(vf, ev))
    lines.append("--- Last events ---")
    for vf, ev in all_pairs[-keep:]:
        lines.append(_ev_line(vf, ev))

    return "\n".join(lines)


class QAEngine:
    def __init__(self, db):
        self.settings = get_settings()
        self.logger = get_logger()
        self.db = db
        self.caption_retriever = CaptionRetriever(db)
        self.object_retriever = ObjectRetriever(db)
        self.repo = EventRepository(db)
        self.model = self.settings.text_model

    def ask(self, question, video_filename=None, camera_id=None,
            min_second=None, max_second=None):
        self.logger.info("qa_engine_asked", question=question, model=self.model)
        if self._should_use_detection_pipeline(video_filename):
            return self._ask_detection(question, video_filename, camera_id,
                                       min_second, max_second)
        return self._ask_captions(question, video_filename, camera_id,
                                  min_second, max_second)

    def _should_use_detection_pipeline(self, video_filename=None):
        if video_filename:
            return self.repo.has_detection_data(video_filename)
        from app.storage.models import DetectedObject
        return self.db.query(DetectedObject).first() is not None

    def _ask_detection(self, question, video_filename=None, camera_id=None,
                       min_second=None, max_second=None):
        # Semantic search to find relevant videos/tracks
        hits = self.object_retriever.search_track_events(
            query=question, video_filename=video_filename, camera_id=camera_id)
        detection_hits = self.object_retriever.search_detections(
            query=question, video_filename=video_filename, camera_id=camera_id,
            min_second=min_second, max_second=max_second)

        if not hits and not detection_hits:
            return {"answer": "No relevant detection data found to answer this question.",
                    "sources": []}

        involved_videos = set()
        for h in hits:
            involved_videos.add(h["video_filename"])
        for h in detection_hits:
            involved_videos.add(h["video_filename"])

        # ── Layer 1: narrative summary ────────────────────────────────────────
        summary_lines = []
        for vf in sorted(involved_videos):
            s = self.db.query(VideoSummary).filter(
                VideoSummary.video_filename == vf).first()
            if s and s.summary_text:
                summary_lines.append(f"[{vf}]\n{s.summary_text}")
        summary_context = "\n\n".join(summary_lines) if summary_lines else "No summary available."

        # ── Layer 2a: semantic memory graph ───────────────────────────────────
        memory_context = _build_memory_context(self.db, involved_videos)

        # ── Layer 2b: VideoTimeline per-second event spine ────────────────────
        timeline_context = _build_timeline_context(self.db, involved_videos)

        # ── Layer 3: per-track behaviour + motion semantics ───────────────────
        behaviour_context = _build_behaviour_context(self.db, involved_videos)

        # ── Layer 4: raw detection events ─────────────────────────────────────
        raw_events = _build_raw_events_context(self.db, involved_videos, self.logger)

        # Assemble rich context — include non-empty sections only
        context_sections = [raw_events]
        if behaviour_context:
            context_sections.insert(0, behaviour_context)
        if timeline_context:
            context_sections.insert(0, timeline_context)
        if memory_context:
            context_sections.insert(0, memory_context)

        full_context = "\n\n".join(context_sections)

        payload = {
            "model": self.model,
            "system": QA_DETECTION_SYSTEM_PROMPT,
            "prompt": QA_DETECTION_USER_TEMPLATE.format(
                summary=summary_context,
                events=full_context,
                question=question),
            "stream": False,
        }
        response = requests.post(
            f"{self.settings.ollama_host}/api/generate",
            json=payload, timeout=600)
        response.raise_for_status()
        answer = response.json()["response"]
        self.logger.info("qa_engine_answered_detection", model=self.model)
        return {"answer": answer, "sources": hits[:6] + detection_hits[:2]}

    def _ask_captions(self, question, video_filename=None, camera_id=None,
                      min_second=None, max_second=None):
        hits = self.caption_retriever.search(
            query=question, video_filename=video_filename, camera_id=camera_id,
            min_second=min_second, max_second=max_second)

        if not hits:
            return {"answer": "No relevant video content found to answer this question.",
                    "sources": []}

        involved_videos = list({h["video_filename"] for h in hits})
        timeline_captions = []
        for vf in involved_videos:
            rows = (self.caption_retriever.db.query(Caption)
                    .filter(Caption.video_filename == vf)
                    .order_by(asc(Caption.frame_second_offset)).all())
            for r in rows:
                timeline_captions.append((vf, r.frame_second_offset, r.caption_text))

        timeline_captions.sort(key=lambda x: (x[0], x[1]))
        context = "\n".join(
            f"[{vf} @ {sec:.1f}s] {_condense_caption(cap)}"
            for vf, sec, cap in timeline_captions)

        payload = {
            "model": self.model,
            "system": QA_SYSTEM_PROMPT,
            "prompt": QA_USER_TEMPLATE.format(captions=context, question=question),
            "stream": False,
        }
        response = requests.post(
            f"{self.settings.ollama_host}/api/generate",
            json=payload, timeout=600)
        response.raise_for_status()
        answer = response.json()["response"]
        self.logger.info("qa_engine_answered_captions", model=self.model)
        return {"answer": answer, "sources": hits}