import os

from memory_agent.eval.core.llm_client import LLMClient


class VanillaRAGAgent:
    """Baseline: retrieve raw dialogue turns with sentence-transformer embeddings."""

    def __init__(self, top_k: int = 5):
        self.llm = LLMClient()
        from sentence_transformers import SentenceTransformer

        model_name = os.getenv("EMBED_MODEL", "models/bge-m3")
        device = os.getenv("EMBED_DEVICE", "cpu")
        self.embed_model = SentenceTransformer(model_name, device=device)
        self.top_k = top_k
        self.chunks: list[str] = []
        self.embeddings = None
        self.last_retrieved: list[str] = []

    def ingest(self, conversation: dict) -> None:
        self.chunks = []
        for session in conversation.get("sessions", []):
            for turn in session.get("turns", []):
                self.chunks.append(f"[{session['date_time']}] {turn['speaker']}: {turn['text']}")
        self._build_index()

    def answer(self, question: str) -> str:
        self.last_retrieved = self._retrieve(question, self.top_k)
        prompt = (
            "You are answering a question about a past conversation. "
            "Use only the retrieved dialogue snippets below. Keep the answer short. "
            "If the snippets do not contain the answer, reply 'unknown'.\n\n"
            f"=== Retrieved snippets ===\n{chr(10).join(self.last_retrieved)}\n\n"
            f"=== Question ===\n{question}\n\n"
            "=== Answer ==="
        )
        return self.llm.tracked_generate(prompt, max_tokens=64).strip()

    def get_trace(self) -> dict:
        return {
            "baseline": "vanilla_rag",
            "retrieved": self.last_retrieved,
            "llm": self.llm.snapshot(),
        }

    def _build_index(self) -> None:
        import numpy as np

        vecs = self.embed_model.encode(self.chunks, normalize_embeddings=True)
        self.embeddings = np.array(vecs, dtype=np.float32)

    def _retrieve(self, query: str, k: int) -> list[str]:
        if not self.chunks:
            return []
        import numpy as np

        qvec = self.embed_model.encode([query], normalize_embeddings=True)[0]
        sims = self.embeddings @ qvec.astype(np.float32)
        idx = np.argsort(-sims)[:k]
        return [self.chunks[i] for i in idx]
