import requests
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.core.logging import get_logger
from app.rag.retriever import CaptionRetriever
from app.prompts.qa_prompt import QA_SYSTEM_PROMPT, QA_USER_TEMPLATE


class QAEngine:
    """
    Full RAG pipeline:
      1. Retrieve top-K relevant captions via semantic search
      2. Format them as context
      3. Ask Ollama to answer the user's question grounded in that context
    """

    def __init__(self, db: Session):
        self.settings = get_settings()
        self.logger = get_logger()
        self.retriever = CaptionRetriever(db)
        self.model = self.settings.multimodal_model

    def ask(
        self,
        question: str,
        video_filename: str = None,
        camera_id: str = None,
        min_second: float = None,
        max_second: float = None,
    ) -> dict:
        """
        Answer a natural language question grounded in retrieved captions.
        Returns the answer plus the supporting captions used as context.
        """
        self.logger.info("qa_engine_asked", question=question)

        # Step 1 — Retrieve relevant captions
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

        # Step 2 — Format context block
        context_lines = [
            f"[{hit['video_filename']} @ {hit['second']:.1f}s] {hit['caption']}"
            for hit in hits
        ]
        context = "\n".join(context_lines)

        # Step 3 — Build prompt
        user_message = QA_USER_TEMPLATE.format(
            captions=context,
            question=question,
        )

        payload = {
            "model": self.model,
            "system": QA_SYSTEM_PROMPT,
            "prompt": user_message,
            "stream": False,
        }

        response = requests.post(
            f"{self.settings.ollama_host}/api/generate",
            json=payload,
            timeout=120,
        )
        response.raise_for_status()

        answer = response.json()["response"]

        self.logger.info("qa_engine_answered", question=question)

        return {
            "answer": answer,
            "sources": hits,
        }