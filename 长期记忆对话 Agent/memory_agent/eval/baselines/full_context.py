from memory_agent.eval.core.llm_client import LLMClient


class FullContextAgent:
    """Baseline: put truncated raw dialogue history into the prompt."""

    def __init__(self, max_chars: int = 12000):
        self.llm = LLMClient()
        self.max_chars = max_chars
        self.history_text = ""

    def ingest(self, conversation: dict) -> None:
        lines = []
        for session in conversation.get("sessions", []):
            lines.append(f"[Session {session['session_id']} @ {session['date_time']}]")
            for turn in session.get("turns", []):
                lines.append(f"{turn['speaker']}: {turn['text']}")
        history = "\n".join(lines)
        self.history_text = history[-self.max_chars :]

    def answer(self, question: str) -> str:
        prompt = (
            "You are an assistant with access to a long conversation between two people. "
            "Answer using only the conversation. Keep the answer short. "
            "If the conversation does not contain the answer, reply 'unknown'.\n\n"
            f"=== Conversation ===\n{self.history_text}\n\n"
            f"=== Question ===\n{question}\n\n"
            "=== Answer ==="
        )
        return self.llm.tracked_generate(prompt, max_tokens=64).strip()

    def get_trace(self) -> dict:
        return {"baseline": "full_context", "llm": self.llm.snapshot()}
