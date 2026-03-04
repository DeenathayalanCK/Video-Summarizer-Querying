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


def _ev_line(vf, ev):
    ad = _build_attr_str(ev)
    obj_desc = f"{ad} {ev.object_class}".strip() if ad else ev.object_class
    return (
        f"[{vf} @ {_fmt_sec(ev.first_seen_second)}-{_fmt_sec(ev.last_seen_second)}] "
        f"{ev.event_type.upper()}: {obj_desc} track #{ev.track_id} "
        f"(duration: {ev.duration_seconds:.0f}s, conf: {ev.best_confidence or 0:.0%})"
    )


def _build_context(db, involved_videos, logger):
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
            f"(avg {sum(durs) / len(durs):.0f}s), "
            f"span {_fmt_sec(group[0].first_seen_second)}"
            f"-{_fmt_sec(group[-1].last_seen_second)}"
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

        # Narrative context from stored video summary
        summary_lines = []
        for vf in sorted(involved_videos):
            s = self.db.query(VideoSummary).filter(
                VideoSummary.video_filename == vf).first()
            if s and s.summary_text:
                summary_lines.append(f"[{vf}]\n{s.summary_text}")
        summary_context = ("\n\n".join(summary_lines)
                           if summary_lines else "No summary available.")

        # Detection timeline with context guard
        context = _build_context(self.db, involved_videos, self.logger)

        payload = {
            "model": self.model,
            "system": QA_DETECTION_SYSTEM_PROMPT,
            "prompt": QA_DETECTION_USER_TEMPLATE.format(
                summary=summary_context, events=context, question=question),
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