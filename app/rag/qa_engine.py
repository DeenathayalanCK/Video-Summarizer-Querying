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
import time as _time
from collections import defaultdict
from sqlalchemy.orm import Session
from sqlalchemy import asc

from app.core.config import get_settings
from app.core.logging import get_logger
from app.core.ollama_logger import OllamaCallTimer, log_call
from app.rag.retriever import CaptionRetriever
from app.rag.object_retriever import ObjectRetriever
from app.rag.temporal_retriever import TemporalRetriever
from app.rag.context_budget import build_budgeted_context, OLLAMA_NUM_CTX, trim_to_budget
from app.rag.fast_path import try_fast_path
from app.storage.repository import EventRepository
from app.storage.models import Caption, TrackEvent, VideoSummary
from app.prompts.qa_prompt import (
    QA_SYSTEM_PROMPT,
    QA_USER_TEMPLATE,
    QA_DETECTION_SYSTEM_PROMPT,
    QA_DETECTION_USER_TEMPLATE,
    FAST_PATH_CURATE_SYSTEM,
    FAST_PATH_CURATE_TEMPLATE,
)

_QA_MAX_EVENTS = 150


def _condense_caption(caption_text: str) -> str:
    """Trim verbose caption prefixes for token efficiency. Inlined to avoid cross-module import."""
    import re as _re
    text = caption_text.strip()
    text = _re.sub(r"(?i)^(the image shows?|this (image|photo|frame|video frame|picture) shows?|"
                   r"in (this|the) (image|photo|frame|video frame|picture)[,:]?)\s*", "", text)
    return text.strip() or caption_text.strip()

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


def _build_caption_context(db, involved_videos: set, max_captions: int = 20) -> str:
    """
    Pull scene captions for involved videos and moondream_raw descriptions
    from TrackEvent attributes.  Used as supplementary context for the LLM.

    Two sources combined:
      1. Caption table — batch-mode scene captions (timestamped scene descriptions)
      2. TrackEvent.attributes['moondream_raw'] — live-mode crop descriptions
         generated by moondream during attribute extraction

    Caps at max_captions total lines to respect context budget.
    """
    lines = ["=== SCENE CAPTIONS & VISUAL DESCRIPTIONS ==="]
    found_any = False

    for vf in sorted(involved_videos):
        # Source 1: batch captions from Caption table
        caps = (
            db.query(Caption)
            .filter(Caption.video_filename == vf)
            .order_by(Caption.frame_second_offset)
            .limit(max_captions)
            .all()
        )
        for c in caps:
            found_any = True
            lines.append(
                f"[{vf} @ {_fmt_sec(c.frame_second_offset)}] {_condense_caption(c.caption_text)}"
            )

        # Source 2: moondream_raw from TrackEvent attributes (live mode)
        entry_evs = (
            db.query(TrackEvent)
            .filter(
                TrackEvent.video_filename == vf,
                TrackEvent.event_type == "entry",
            )
            .order_by(asc(TrackEvent.first_seen_second))
            .all()
        )
        added = 0
        for ev in entry_evs:
            raw = (ev.attributes or {}).get("moondream_raw", "").strip()
            if not raw:
                continue
            found_any = True
            lines.append(
                f"[{vf} Track #{ev.track_id} @ {_fmt_sec(ev.first_seen_second)}] "
                f"Visual: {raw[:300]}"
            )
            added += 1
            if added >= max_captions:
                break

    if not found_any:
        return ""
    return "\n".join(lines)


class QAEngine:
    def __init__(self, db):
        self.settings = get_settings()
        self.logger = get_logger()
        self.db = db
        self.ask_timeout_s = max(30, self.settings.ask_timeout_seconds)
        self.caption_retriever = CaptionRetriever(db, top_k=max(1, self.settings.qa_top_k_object))
        self.object_retriever = ObjectRetriever(db, top_k=max(1, self.settings.qa_top_k_object))
        self.temporal_retriever = TemporalRetriever(db, top_k=max(1, self.settings.qa_top_k_temporal))
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

    @staticmethod
    def _is_already_clean(raw_facts: str) -> bool:
        """
        Returns True when raw_facts is already a clean, readable sentence and
        does NOT need to be sent to the LLM for curation.

        Criteria:
        - Single line (no newlines)
        - No markdown bold (**) or backticks  (those indicate structured DB output)
        - Under 160 chars (a full natural-language sentence)

        This skips the Ollama curate call entirely for simple "No X detected"
        answers, avoiding the 25s timeout on an already-clean result.
        """
        stripped = raw_facts.strip()
        return (
            "\n" not in stripped
            and "**" not in stripped
            and "`" not in stripped
            and len(stripped) < 160
        )

    def _curate_fast_path_sync(self, raw_facts: str, question: str) -> str:
        """
        Synchronous version of the fast-path curation step used by _ask_detection.

        Sends the raw DB facts + user question to the LLM with a tiny prompt
        (~200 tokens) so it rewrites them into a clean, natural-language answer.
        Falls back to a summarised form of raw_facts if Ollama is unavailable or
        times out — never dumps verbose multi-line raw data directly to the user.

        Safety cap: raw_facts is truncated to 600 chars before building the
        curate prompt.  Time-filtered queries should never produce more than a
        few lines; if they do (e.g. a broad "all videos" count with no time
        range) we still cap rather than send thousands of tokens that will OOM
        the model or time out.
        """
        # Skip curate entirely if raw_facts is already a clean readable sentence
        # (e.g. "No person were detected between 10:30–10:40.")
        if self._is_already_clean(raw_facts):
            self.logger.info("fast_path_curate_skipped_already_clean")
            return raw_facts.strip()

        # Hard cap on raw_facts fed to curate — prevents timeout on large blobs
        _MAX_RAW = 600
        raw_facts_capped = raw_facts[:_MAX_RAW]
        if len(raw_facts) > _MAX_RAW:
            raw_facts_capped += f"\n... ({len(raw_facts) - _MAX_RAW} chars truncated)"
            self.logger.warning(
                "fast_path_curate_raw_facts_capped",
                original_len=len(raw_facts),
                cap=_MAX_RAW,
            )

        curate_prompt = FAST_PATH_CURATE_TEMPLATE.format(
            raw_facts=raw_facts_capped,
            question=question,
        )
        try:
            with OllamaCallTimer(
                call_type="ask",
                model=self.model,
                prompt=curate_prompt[:800],
            ) as _ct:
                resp = requests.post(
                    f"{self.settings.ollama_host}/api/generate",
                    json={
                        "model": self.model,
                        "system": FAST_PATH_CURATE_SYSTEM,
                        "prompt": curate_prompt,
                        "stream": False,
                        "options": {"num_ctx": 512, "num_predict": 150},
                    },
                    timeout=(10, 25),   # curate prompt is tiny; >25s means Ollama is stuck
                )
                resp.raise_for_status()
                curated = resp.json().get("response", "").strip()
                _ct.response = curated[:500]
                if curated:
                    return curated
        except Exception as _e:
            self.logger.warning("fast_path_curate_sync_failed", error=str(_e))
        # Fallback: return first 2 lines of raw_facts (not the whole blob)
        first_lines = "\n".join(raw_facts.splitlines()[:3])
        return first_lines if first_lines else raw_facts[:200]

    def _ask_detection(self, question, video_filename=None, camera_id=None,
                       min_second=None, max_second=None):

        # ── Fast-path: answer from DB, then curate via LLM ──────────────────
        # DB query gives raw structured facts in <100ms; the LLM curate step
        # rewrites them into a natural-language answer (~3-8s, tiny prompt).
        # This is far faster than the full detection pipeline (2-7 min) while
        # still giving the user a clean, readable response with evidence.
        fast = try_fast_path(self.db, question, video_filename, min_second, max_second)
        if fast.get("answered"):
            self.logger.info(
                "qa_fast_path_answered",
                fast_path_type=fast.get("fast_path_type"),
                question=question,
            )
            curated = self._curate_fast_path_sync(
                raw_facts=fast["answer"],
                question=question,
            )
            return {
                "answer": curated,
                "sources": fast.get("sources", []),
                "fast_path": True,
                "fast_path_type": fast.get("fast_path_type"),
                "db_evidence": {
                    "raw_answer": fast["answer"],
                    "fast_path_type": fast.get("fast_path_type"),
                    "sources": fast.get("sources", []),
                },
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

        # Guard: for "All videos" broad queries, cap to the most recent 8 windows.
        # Querying 15+ windows blows the context budget even after trimming because
        # the number of layers (memory, timeline, behaviour) scales with window count.
        # The retriever already ranks by relevance — the most relevant windows come
        # first; additional old windows add noise more than signal.
        if not video_filename and len(involved_videos) > 8:
            sorted_vids = sorted(involved_videos, reverse=True)  # lexicographic = newest first
            involved_videos = set(sorted_vids[:8])
            self.logger.info("qa_involved_videos_capped", total=len(sorted_vids), kept=8)

        if not involved_videos:
            return {
                "answer": "No relevant detection data found to answer this question.",
                "sources": [],
            }

        # ── Layer 1: narrative summary (BUDGETED) ────────────────────────────
        # IMPORTANT: summary_context is injected into the prompt alongside
        # build_budgeted_context output. It MUST be capped independently or it
        # blows past num_ctx=2048 (10 windows × ~300 tokens = 3000 extra tokens).
        # Cap each summary to ~1 sentence (80 tokens) and total to 500 tokens.
        summary_lines = []
        for vf in sorted(involved_videos):
            s = self.db.query(VideoSummary).filter(
                VideoSummary.video_filename == vf).first()
            if s and s.summary_text:
                # Take only first 2 sentences of each summary
                sentences = s.summary_text.replace("\n", " ").split(". ")
                short = ". ".join(sentences[:2]).strip()
                if short and not short.endswith("."):
                    short += "."
                summary_lines.append(f"[{vf}] {short}")
        raw_summary = "\n".join(summary_lines) if summary_lines else ""

        # ── Focused track context (Pass 1 output) ─────────────────────────────
        focused_context = self.temporal_retriever.render_contexts_as_text(track_contexts)

        # ── Broad context layers (Pass 2) ─────────────────────────────────────
        memory_context    = _build_memory_context(self.db, involved_videos)
        timeline_context  = _build_timeline_context(self.db, involved_videos)
        behaviour_context = _build_behaviour_context(self.db, involved_videos)
        raw_events        = _build_raw_events_context(self.db, involved_videos, self.logger)
        caption_context   = _build_caption_context(self.db, involved_videos)

        full_context = build_budgeted_context(
            focused=focused_context,
            summary=raw_summary,
            memory=memory_context,
            behaviour=behaviour_context,
            timeline=timeline_context,
            raw_events=raw_events,
            captions=caption_context,
        )

        prompt = QA_DETECTION_USER_TEMPLATE.format(
            summary="(included in context below)",
            events=full_context,
            question=question)

        with OllamaCallTimer(call_type="ask", model=self.model, prompt=prompt[:800]) as _ot:
            response = requests.post(
                f"{self.settings.ollama_host}/api/generate",
                json={
                    "model": self.model,
                    "system": QA_DETECTION_SYSTEM_PROMPT,
                    "prompt": prompt,
                    "stream": False,
                    "options": {"num_ctx": OLLAMA_NUM_CTX, "num_predict": 200},
                },
                timeout=(10, self.ask_timeout_s))
            response.raise_for_status()
            answer = response.json()["response"]
            _ot.response = answer

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
        prompt = QA_USER_TEMPLATE.format(captions=context, question=question)
        with OllamaCallTimer(call_type="ask_captions", model=self.model, prompt=prompt[:800]) as _ot:
            response = requests.post(
                f"{self.settings.ollama_host}/api/generate",
                json={
                    "model": self.model,
                    "system": QA_SYSTEM_PROMPT,
                    "prompt": prompt,
                    "stream": False,
                    "options": {"num_ctx": OLLAMA_NUM_CTX, "num_predict": 200},
                },
                timeout=(10, self.ask_timeout_s))
            response.raise_for_status()
            answer = response.json()["response"]
            _ot.response = answer
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
        fast = try_fast_path(self.db, question, video_filename, min_second, max_second)
        if fast.get("answered"):
            self.logger.info("qa_stream_fast_path", type=fast.get("fast_path_type"))
            raw_answer  = fast["answer"]
            raw_sources = fast.get("sources", [])
            fp_type     = fast.get("fast_path_type", "db")

            # ── Curate: send raw DB facts + question to LLM ──────────────────
            # Skip curate entirely if raw_facts is already a clean readable sentence
            # (single line, no markdown, <160 chars). Avoids 25s Ollama timeout
            # for simple "No X detected" answers that need no rewriting.
            t_curate_start = _time.monotonic()   # defined here for elapsed calc below
            if self._is_already_clean(raw_answer):
                self.logger.info("stream_fast_path_curate_skipped_already_clean")
                yield _sse({"token": raw_answer.strip()})
            else:
                # Hard cap on raw_facts fed to curate — prevents timeout on large blobs.
                _MAX_RAW = 600
                raw_answer_capped = raw_answer[:_MAX_RAW]
                if len(raw_answer) > _MAX_RAW:
                    raw_answer_capped += f"\n... ({len(raw_answer) - _MAX_RAW} chars truncated)"
                    self.logger.warning(
                        "stream_curate_raw_facts_capped",
                        original_len=len(raw_answer),
                        cap=_MAX_RAW,
                    )

                curate_prompt = FAST_PATH_CURATE_TEMPLATE.format(
                    raw_facts=raw_answer_capped,
                    question=question,
                )
                curated_tokens = []
                try:
                    with OllamaCallTimer(
                        call_type="ask",
                        model=self.model,
                        prompt=curate_prompt[:800],
                    ) as _ct:
                        resp = requests.post(
                            f"{self.settings.ollama_host}/api/generate",
                            json={
                                "model": self.model,
                                "system": FAST_PATH_CURATE_SYSTEM,
                                "prompt": curate_prompt,
                                "stream": True,
                                "options": {"num_ctx": 512, "num_predict": 150},
                            },
                            stream=True,
                            timeout=(10, 25),   # curate prompt is tiny; >25s means Ollama is stuck
                        )
                        resp.raise_for_status()
                        import json as _jfp
                        for line in resp.iter_lines():
                            if not line:
                                continue
                            try:
                                chunk = _jfp.loads(line)
                            except Exception:
                                continue
                            token = chunk.get("response", "")
                            if token:
                                curated_tokens.append(token)
                                yield _sse({"token": token})
                            if chunk.get("done"):
                                break
                        _ct.response = "".join(curated_tokens)[:500]
                except Exception as _fe:
                    self.logger.warning("fast_path_curate_failed", error=str(_fe))
                    # Fallback: stream first 3 lines only (not the full raw blob)
                    fallback = "\n".join(raw_answer.splitlines()[:3])
                    yield _sse({"token": fallback if fallback else raw_answer[:200]})

            curate_elapsed_ms = (_time.monotonic() - t_curate_start) * 1000

            # ── Done: include raw DB data as db_evidence for the UI ───────────
            yield _sse({
                "done":           True,
                "sources":        raw_sources,
                "fast_path":      True,
                "fast_path_type": fp_type,
                "db_evidence": {
                    "raw_answer": raw_answer,
                    "fast_path_type": fp_type,
                    "sources": raw_sources,
                },
                "elapsed_ms":     round(curate_elapsed_ms),
            })
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
            yield _sse({"done": True, "sources": [], "elapsed_ms": 0, "context": ""})
            return

        summary_lines = []
        for vf in sorted(involved_videos):
            s = self.db.query(VideoSummary).filter(
                VideoSummary.video_filename == vf).first()
            if s and s.summary_text:
                sentences = s.summary_text.replace("\n", " ").split(". ")
                short = ". ".join(sentences[:2]).strip()
                if short and not short.endswith("."):
                    short += "."
                summary_lines.append(f"[{vf}] {short}")
        raw_summary = "\n".join(summary_lines) if summary_lines else ""

        focused_context   = self.temporal_retriever.render_contexts_as_text(track_contexts)
        memory_context    = _build_memory_context(self.db, involved_videos)
        timeline_context  = _build_timeline_context(self.db, involved_videos)
        behaviour_context = _build_behaviour_context(self.db, involved_videos)
        raw_events        = _build_raw_events_context(self.db, involved_videos, self.logger)
        caption_context   = _build_caption_context(self.db, involved_videos)

        full_context = build_budgeted_context(
            focused=focused_context, summary=raw_summary, memory=memory_context,
            behaviour=behaviour_context, timeline=timeline_context,
            raw_events=raw_events, captions=caption_context,
        )

        prompt = QA_DETECTION_USER_TEMPLATE.format(
            summary="(included in context below)", events=full_context, question=question)

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
        full_response_tokens = []
        llm_elapsed_ms = 0.0
        t_llm_start = _time.monotonic()
        try:
            with OllamaCallTimer(
                call_type="ask",
                model=self.model,
                prompt=prompt[:800],
            ) as _ot:
                resp = requests.post(
                    f"{self.settings.ollama_host}/api/generate",
                    json={
                        "model": self.model,
                        "system": QA_DETECTION_SYSTEM_PROMPT,
                        "prompt": prompt,
                        "stream": True,
                        "options": {"num_ctx": OLLAMA_NUM_CTX, "num_predict": 200},
                    },
                    stream=True,
                    timeout=(10, self.ask_timeout_s),
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
                        full_response_tokens.append(token)
                        yield _sse({"token": token})
                    if chunk.get("done"):
                        break

                _ot.response = "".join(full_response_tokens)[:500]

        except Exception as e:
            self.logger.error("qa_stream_error", error=str(e))
            yield _sse({"token": f"\n\n[Error during generation: {e}]"})

        llm_elapsed_ms = (_time.monotonic() - t_llm_start) * 1000

        # ── Context section sizes for frontend debug panel ────────────────────
        context_sections = {
            "focused_tracks": len(focused_context),
            "captions": len(caption_context),
            "memory": len(memory_context),
            "timeline": len(timeline_context),
            "behaviour": len(behaviour_context),
            "raw_events": len(raw_events),
            "summary": len(raw_summary),
            "total_chars": len(full_context),
        }

        yield _sse({
            "done": True,
            "sources": sources,
            "elapsed_ms": round(llm_elapsed_ms),
            "context": full_context,
            "context_sections": context_sections,
        })