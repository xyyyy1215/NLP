from memory_agent.eval.core.llm_client import LLMClient


class NoMemoryAgent:
    """Baseline: answer with only the current question."""

    def __init__(self):
        self.llm = LLMClient()

    def ingest(self, conversation: dict) -> None:
        self.conversation_meta = {
            "speaker_a": conversation.get("speaker_a"),
            "speaker_b": conversation.get("speaker_b"),
            "sessions": len(conversation.get("sessions", [])),
        }

    def answer(self, question: str) -> str:
        prompt = (
            "Answer the question briefly. If you do not know the answer, reply 'unknown'.\n\n"
            f"Question: {question}\nAnswer:"
        )
        return self.llm.tracked_generate(prompt, max_tokens=64).strip()

    def get_trace(self) -> dict:
        return {"baseline": "no_memory", "llm": self.llm.snapshot()}

