"""
qa_engine.py — Temporal-aware Q&A over video surveillance data.

Context is now built in TWO passes:

  PASS 1 — FOCUSED (TemporalRetriever)
  ------------------------------------
  Detects query intent, retrieves relevant tracks with FULL temporal context.
  Groups events by track so LLM sees complete timeline per physical object.
  For temporal queries ("fell after walking") — expands to the full track window.
  For behaviour queries — searches memory graph directly for behaviour labels.

  PASS 2 — BROAD (legacy 4-layer fallback)
  ----------------------------------------
  Narrative summary + remaining memory nodes + behaviour analysis + raw events.
  Used when focused retrieval alone is insufficient or as background context.

Final prompt structure:
  [VIDEO SUMMARY]
  [FOCUSED TRACK CONTEXT]   ← new: track-grouped, memory-enriched, temporal
  [SEMANTIC MEMORY GRAPH]
  [VIDEO TIMELINE]
  [BEHAVIOUR ANALYSIS]
  [RAW DETECTION EVENTS]
"""

import requests
from collections import defaultdict
from sqlalchemy.orm import Session
from sqlalchemy import asc

from app.core.config import get_settings
from app.core.logging import get_logger
from app.rag.retriever import CaptionRetriever
from app.rag.object_retriever import ObjectRetriever
from app.rag.temporal_retriever import TemporalRetriever
from app.rag.context_budget import build_budgeted_context, OLLAMA_NUM_CTX
from app.rag.fast_path import try_fast_path
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
        plate = attrs.get("plate_number", "")
        if plate and plate not in ("unknown", ""):
            parts.append(f"plate:{plate}")
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
    lines = ["=== BEHAVIOUR ANALYSIS (per track) ==="]
    found_any = False
    for vf in sorted(involved_videos):
        entry_evs = (
            db.query(TrackEvent)
            .filter(TrackEvent.video_filename == vf, TrackEvent.event_type == "entry")
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
        return ""
    return "\n".join(lines)


def _build_timeline_context(db, involved_videos: set, max_entries: int = 80) -> str:
    from app.storage.models import VideoTimeline
    lines = ["=== VIDEO TIMELINE (per-second events) ==="]
    found_any = False
    for vf in sorted(involved_videos):
        tl = db.query(VideoTimeline).filter(VideoTimeline.video_filename == vf).first()
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
    from app.storage.models import SemanticMemoryGraph
    lines = ["=== SEMANTIC MEMORY (knowledge graph) ==="]
    found_any = False
    for vf in sorted(involved_videos):
        graphs = (
            db.query(SemanticMemoryGraph)
            .filter(SemanticMemoryGraph.video_filename == vf)
            .order_by(SemanticMemoryGraph.node_type, SemanticMemoryGraph.track_id)
            .limit(40)   # cap: 40 nodes * ~25 tokens = ~1000 tokens max
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

    logger.info("qa_context_guard_triggered", total=len(all_pairs), cap=_QA_MAX_EVENTS)
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
        self.temporal_retriever = TemporalRetriever(db)
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

        # ── Fast-path: answer directly from DB if possible ────────────────────
        # Skips LLM entirely for factual queries (plate, count, presence, etc.)
        # Response time: <100ms vs 2-7 minutes for LLM path
        fast = try_fast_path(self.db, question, video_filename)
        if fast.get("answered"):
            self.logger.info(
                "qa_fast_path_answered",
                fast_path_type=fast.get("fast_path_type"),
                question=question,
            )
            return {
                "answer": fast["answer"],
                "sources": fast.get("sources", []),
                "fast_path": True,
                "fast_path_type": fast.get("fast_path_type"),
            }

        # ── PASS 1: Temporal / focused retrieval ──────────────────────────────
        intent = self.temporal_retriever.parse_temporal_intent(question)
        self.logger.info(
            "qa_temporal_intent",
            is_temporal=intent["is_temporal"],
            keywords=intent["keywords_found"],
            sequence=intent.get("sequence_hint"),
        )

        if intent["is_temporal"]:
            track_contexts = self.temporal_retriever.retrieve_temporal(
                question, video_filename, intent=intent)
        else:
            track_contexts = self.temporal_retriever.retrieve_track_grouped(
                question, video_filename)

        # Derive involved_videos from track contexts + legacy fallback search
        involved_videos = {ctx.video_filename for ctx in track_contexts}

        # Legacy semantic search for involved_videos discovery (still needed
        # for broad queries where temporal retriever finds nothing)
        legacy_hits = self.object_retriever.search_track_events(
            query=question, video_filename=video_filename, camera_id=camera_id)
        detection_hits = self.object_retriever.search_detections(
            query=question, video_filename=video_filename, camera_id=camera_id,
            min_second=min_second, max_second=max_second)

        for h in legacy_hits:
            involved_videos.add(h["video_filename"])
        for h in detection_hits:
            involved_videos.add(h["video_filename"])

        if not involved_videos:
            return {
                "answer": "No relevant detection data found to answer this question.",
                "sources": [],
            }

        # ── Layer 1: narrative summary ────────────────────────────────────────
        summary_lines = []
        for vf in sorted(involved_videos):
            s = self.db.query(VideoSummary).filter(
                VideoSummary.video_filename == vf).first()
            if s and s.summary_text:
                summary_lines.append(f"[{vf}]\n{s.summary_text}")
        summary_context = "\n\n".join(summary_lines) if summary_lines else "No summary available."

        # ── Focused track context (Pass 1 output) ─────────────────────────────
        focused_context = self.temporal_retriever.render_contexts_as_text(track_contexts)

        # ── Broad context layers (Pass 2) ─────────────────────────────────────
        memory_context   = _build_memory_context(self.db, involved_videos)
        timeline_context = _build_timeline_context(self.db, involved_videos)
        behaviour_context = _build_behaviour_context(self.db, involved_videos)
        raw_events       = _build_raw_events_context(self.db, involved_videos, self.logger)

        # Assemble context within strict token budget to prevent Ollama timeout
        # Each section is trimmed to its allocation; total guaranteed < num_ctx
        full_context = build_budgeted_context(
            focused=focused_context,
            memory=memory_context,
            behaviour=behaviour_context,
            timeline=timeline_context,
            raw_events=raw_events,
        )

        payload = {
            "model": self.model,
            "system": QA_DETECTION_SYSTEM_PROMPT,
            "prompt": QA_DETECTION_USER_TEMPLATE.format(
                summary=summary_context,
                events=full_context,
                question=question),
            "stream": False,
            "options": {
                "num_ctx": OLLAMA_NUM_CTX,   # explicitly set context window
                "num_predict": 512,           # cap response length — answers don't need 4096 tokens
            },
        }
        response = requests.post(
            f"{self.settings.ollama_host}/api/generate",
            json=payload, timeout=600)
        response.raise_for_status()
        answer = response.json()["response"]

        self.logger.info(
            "qa_engine_answered_detection",
            model=self.model,
            is_temporal=intent["is_temporal"],
            track_contexts=len(track_contexts),
        )

        # Sources: track contexts + legacy hits
        sources = []
        for ctx in track_contexts:
            sources.append({
                "video_filename": ctx.video_filename,
                "track_id": ctx.track_id,
                "object_class": ctx.object_class,
                "first_seen": ctx.first_seen,
                "last_seen": ctx.last_seen,
                "best_crop_path": ctx.best_crop_path,
                "retrieval_reason": ctx.retrieval_reason,
                "relevance_score": ctx.relevance_score,
            })
        sources.extend(legacy_hits[:4])
        return {"answer": answer, "sources": sources}

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
            "options": {
                "num_ctx": OLLAMA_NUM_CTX,
                "num_predict": 512,
            },
        }
        response = requests.post(
            f"{self.settings.ollama_host}/api/generate",
            json=payload, timeout=600)
        response.raise_for_status()
        answer = response.json()["response"]
        self.logger.info("qa_engine_answered_captions", model=self.model)
        return {"answer": answer, "sources": hits}

# stream_ask method placeholder — will be added cleanly

    def stream_ask(self, question, video_filename=None, camera_id=None,
                   min_second=None, max_second=None):
        """
        Generator that streams the answer token-by-token from Ollama.
        Yields SSE-formatted strings:
          data: {"token": "..."}\n\n   <- one per Ollama chunk
          data: {"done": true, "sources": [...], ...}\n\n  <- final frame

        Fast-path answers are emitted as a single token + done (no LLM call).
        """
        import json as _json

        def _sse(obj):
            return f"data: {_json.dumps(obj)}\n\n"

        # ── Fast-path: answer from DB in <100ms ───────────────────────────────
        fast = try_fast_path(self.db, question, video_filename)
        if fast.get("answered"):
            self.logger.info("qa_stream_fast_path", type=fast.get("fast_path_type"))
            yield _sse({"token": fast["answer"]})
            yield _sse({"done": True, "sources": fast.get("sources", []),
                        "fast_path": True, "fast_path_type": fast.get("fast_path_type")})
            return

        # ── Caption path (non-detection videos) ───────────────────────────────
        if not self._should_use_detection_pipeline(video_filename):
            result = self._ask_captions(question, video_filename, camera_id,
                                        min_second, max_second)
            yield _sse({"token": result["answer"]})
            yield _sse({"done": True, "sources": result["sources"]})
            return

        # ── Detection path: build context ─────────────────────────────────────
        intent = self.temporal_retriever.parse_temporal_intent(question)
        if intent["is_temporal"]:
            track_contexts = self.temporal_retriever.retrieve_temporal(
                question, video_filename, intent=intent)
        else:
            track_contexts = self.temporal_retriever.retrieve_track_grouped(
                question, video_filename)

        involved_videos = {ctx.video_filename for ctx in track_contexts}
        legacy_hits = self.object_retriever.search_track_events(
            query=question, video_filename=video_filename, camera_id=camera_id)
        detection_hits = self.object_retriever.search_detections(
            query=question, video_filename=video_filename, camera_id=camera_id,
            min_second=min_second, max_second=max_second)
        for h in legacy_hits:
            involved_videos.add(h["video_filename"])
        for h in detection_hits:
            involved_videos.add(h["video_filename"])

        if not involved_videos:
            yield _sse({"token": "No relevant detection data found to answer this question."})
            yield _sse({"done": True, "sources": []})
            return

        summary_lines = []
        for vf in sorted(involved_videos):
            s = self.db.query(VideoSummary).filter(
                VideoSummary.video_filename == vf).first()
            if s and s.summary_text:
                summary_lines.append(f"[{vf}]\n{s.summary_text}")
        summary_context = "\n\n".join(summary_lines) if summary_lines else "No summary available."

        focused_context   = self.temporal_retriever.render_contexts_as_text(track_contexts)
        memory_context    = _build_memory_context(self.db, involved_videos)
        timeline_context  = _build_timeline_context(self.db, involved_videos)
        behaviour_context = _build_behaviour_context(self.db, involved_videos)
        raw_events        = _build_raw_events_context(self.db, involved_videos, self.logger)

        full_context = build_budgeted_context(
            focused=focused_context, memory=memory_context,
            behaviour=behaviour_context, timeline=timeline_context,
            raw_events=raw_events,
        )

        prompt = QA_DETECTION_USER_TEMPLATE.format(
            summary=summary_context, events=full_context, question=question)

        sources = []
        for ctx in track_contexts:
            sources.append({
                "video_filename": ctx.video_filename,
                "track_id": ctx.track_id,
                "object_class": ctx.object_class,
                "first_seen": ctx.first_seen,
                "last_seen": ctx.last_seen,
                "best_crop_path": ctx.best_crop_path,
                "retrieval_reason": ctx.retrieval_reason,
            })
        sources.extend(legacy_hits[:4])

        # ── Stream from Ollama ────────────────────────────────────────────────
        try:
            resp = requests.post(
                f"{self.settings.ollama_host}/api/generate",
                json={
                    "model": self.model,
                    "system": QA_DETECTION_SYSTEM_PROMPT,
                    "prompt": prompt,
                    "stream": True,
                    "options": {"num_ctx": OLLAMA_NUM_CTX, "num_predict": 512},
                },
                stream=True,
                timeout=600,
            )
            resp.raise_for_status()

            import json as _json2
            for line in resp.iter_lines():
                if not line:
                    continue
                try:
                    chunk = _json2.loads(line)
                except Exception:
                    continue
                token = chunk.get("response", "")
                if token:
                    yield _sse({"token": token})
                if chunk.get("done"):
                    break

        except Exception as e:
            self.logger.error("qa_stream_error", error=str(e))
            yield _sse({"token": f"\n\n[Error during generation: {e}]"})

        yield _sse({"done": True, "sources": sources})