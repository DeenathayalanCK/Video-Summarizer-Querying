import requests
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.core.logging import get_logger
from app.rag.retriever import CaptionRetriever
from app.rag.summarizer import _condense_caption
from app.prompts.qa_prompt import QA_SYSTEM_PROMPT, QA_USER_TEMPLATE


class QAEngine:
    def __init__(self, db: Session):
        self.settings = get_settings()
        self.logger = get_logger()
        self.retriever = CaptionRetriever(db)
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

        hits = self.retriever.search(
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

        # For QA, also fetch the full timeline of each involved video
        # so the LLM has sequential context to reason across, not just
        # the top-K semantically similar hits (which may all be similar frames)
        from app.storage.models import Caption
        from sqlalchemy import asc

        involved_videos = list({h["video_filename"] for h in hits})
        timeline_captions = []

        for vf in involved_videos:
            rows = (
                self.retriever.db.query(Caption)
                .filter(Caption.video_filename == vf)
                .order_by(asc(Caption.frame_second_offset))
                .all()
            )
            for r in rows:
                timeline_captions.append((vf, r.frame_second_offset, r.caption_text))

        # Sort by video then time
        timeline_captions.sort(key=lambda x: (x[0], x[1]))

        # Build condensed sequential context
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
        self.logger.info("qa_engine_answered", model=self.model)

        return {"answer": answer, "sources": hits}