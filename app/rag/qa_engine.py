import requests
from sqlalchemy.orm import Session
from sqlalchemy import asc

from app.core.config import get_settings
from app.core.logging import get_logger
from app.rag.retriever import CaptionRetriever
from app.rag.object_retriever import ObjectRetriever
from app.rag.summarizer import _condense_caption
from app.storage.repository import EventRepository
from app.storage.models import Caption, TrackEvent
from app.prompts.qa_prompt import (
    QA_SYSTEM_PROMPT,
    QA_USER_TEMPLATE,
    QA_DETECTION_SYSTEM_PROMPT,
    QA_DETECTION_USER_TEMPLATE,
)


class QAEngine:
    def __init__(self, db: Session):
        self.settings = get_settings()
        self.logger = get_logger()
        self.db = db
        self.caption_retriever = CaptionRetriever(db)
        self.object_retriever = ObjectRetriever(db)
        self.repo = EventRepository(db)
        self.model = self.settings.text_model

    def ask(
        self,
        question: str,
        video_filename: str = None,
        camera_id: str = None,
        min_second: float = None,
        max_second: float = None,
    ) -> dict:
        self.logger.info("qa_engine_asked", question=question, model=self.model)

        # Route to correct pipeline based on available data
        use_detection = self._should_use_detection_pipeline(video_filename)

        if use_detection:
            return self._ask_detection(
                question=question,
                video_filename=video_filename,
                camera_id=camera_id,
                min_second=min_second,
                max_second=max_second,
            )
        else:
            return self._ask_captions(
                question=question,
                video_filename=video_filename,
                camera_id=camera_id,
                min_second=min_second,
                max_second=max_second,
            )

    def _should_use_detection_pipeline(self, video_filename: str = None) -> bool:
        if video_filename:
            return self.repo.has_detection_data(video_filename)
        else:
            from app.storage.models import DetectedObject
            return self.db.query(DetectedObject).first() is not None

    # ── Detection pipeline Q&A (Phase 6A / 6B) ────────────────────────────────

    def _ask_detection(
        self,
        question: str,
        video_filename: str = None,
        camera_id: str = None,
        min_second: float = None,
        max_second: float = None,
    ) -> dict:
        hits = self.object_retriever.search_track_events(
            query=question,
            video_filename=video_filename,
            camera_id=camera_id,
        )
        detection_hits = self.object_retriever.search_detections(
            query=question,
            video_filename=video_filename,
            camera_id=camera_id,
            min_second=min_second,
            max_second=max_second,
        )

        if not hits and not detection_hits:
            return {
                "answer": "No relevant detection data found to answer this question.",
                "sources": [],
            }

        involved_videos = set()
        for h in hits:
            involved_videos.add(h["video_filename"])
        for h in detection_hits:
            involved_videos.add(h["video_filename"])

        timeline_lines = []
        for vf in sorted(involved_videos):
            events = (
                self.db.query(TrackEvent)
                .filter(TrackEvent.video_filename == vf)
                .order_by(asc(TrackEvent.first_seen_second))
                .all()
            )
            for ev in events:
                attr_str = ""
                if ev.attributes:
                    attrs = ev.attributes
                    cls = attrs.get("object_class", ev.object_class)
                    if cls in ("car", "truck", "bus", "motorcycle", "bicycle"):
                        color = attrs.get("color", "")
                        vtype = attrs.get("type", "")
                        make = attrs.get("make_estimate", "")
                        parts = [p for p in [color, vtype] if p and p != "unknown"]
                        if make and make != "unknown":
                            parts.append(f"({make})")
                        attr_str = " ".join(parts)
                    elif cls == "person":
                        gender = attrs.get("gender_estimate", "")
                        top = attrs.get("clothing_top", "")
                        bottom = attrs.get("clothing_bottom", "")
                        parts = [p for p in [gender, top, bottom] if p and p not in ("unknown", "none")]
                        attr_str = ", ".join(parts)

                obj_desc = f"{attr_str} {ev.object_class}".strip() if attr_str else ev.object_class
                timeline_lines.append(
                    f"[{vf} @ {ev.first_seen_second:.1f}s-{ev.last_seen_second:.1f}s] "
                    f"{ev.event_type.upper()}: {obj_desc} track #{ev.track_id} "
                    f"(duration: {ev.duration_seconds:.1f}s, conf: {ev.best_confidence or 0:.0%})"
                )

        context = "\n".join(timeline_lines)

        payload = {
            "model": self.model,
            "system": QA_DETECTION_SYSTEM_PROMPT,
            "prompt": QA_DETECTION_USER_TEMPLATE.format(
                events=context,
                question=question,
            ),
            "stream": False,
        }

        response = requests.post(
            f"{self.settings.ollama_host}/api/generate",
            json=payload,
            timeout=600,
        )
        response.raise_for_status()
        answer = response.json()["response"]
        self.logger.info("qa_engine_answered_detection", model=self.model)

        sources = hits[:6] + detection_hits[:2]
        return {"answer": answer, "sources": sources}

    # ── Caption pipeline Q&A (legacy Phase 5) ─────────────────────────────────

    def _ask_captions(
        self,
        question: str,
        video_filename: str = None,
        camera_id: str = None,
        min_second: float = None,
        max_second: float = None,
    ) -> dict:
        hits = self.caption_retriever.search(
            query=question,
            video_filename=video_filename,
            camera_id=camera_id,
            min_second=min_second,
            max_second=max_second,
        )

        if not hits:
            return {
                "answer": "No relevant video content found to answer this question.",
                "sources": [],
            }

        involved_videos = list({h["video_filename"] for h in hits})
        timeline_captions = []

        for vf in involved_videos:
            rows = (
                self.caption_retriever.db.query(Caption)
                .filter(Caption.video_filename == vf)
                .order_by(asc(Caption.frame_second_offset))
                .all()
            )
            for r in rows:
                timeline_captions.append((vf, r.frame_second_offset, r.caption_text))

        timeline_captions.sort(key=lambda x: (x[0], x[1]))
        context_lines = [
            f"[{vf} @ {sec:.1f}s] {_condense_caption(cap)}"
            for vf, sec, cap in timeline_captions
        ]
        context = "\n".join(context_lines)

        payload = {
            "model": self.model,
            "system": QA_SYSTEM_PROMPT,
            "prompt": QA_USER_TEMPLATE.format(captions=context, question=question),
            "stream": False,
        }

        response = requests.post(
            f"{self.settings.ollama_host}/api/generate",
            json=payload,
            timeout=600,
        )
        response.raise_for_status()
        answer = response.json()["response"]
        self.logger.info("qa_engine_answered_captions", model=self.model)
        return {"answer": answer, "sources": hits}