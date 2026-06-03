from memory_agent.eval.core.llm_client import LLMClient
from memory_agent.memory.retriever import MemoryRetriever
from memory_agent.memory.store import MemoryStore
from memory_agent.memory.updater import MemoryUpdater
from memory_agent.memory.writer import MemoryWriter


class MemoryAgent:
    """Main long-term memory agent: write memories, retrieve them, then answer."""

    def __init__(self, top_k: int = 6):
        self.llm = LLMClient()
        self.store = MemoryStore()
        self.writer = MemoryWriter(self.llm)
        self.updater = MemoryUpdater()
        self.retriever = MemoryRetriever(self.store, top_k=top_k)
        self.last_trace: dict = {}

    def ingest(self, conversation: dict) -> None:
        raw_memories = self.writer.write(conversation)
        memories = self.updater.merge(raw_memories)
        for memory in memories:
            self.store.add(memory)
        self.store.rebuild()
        self.last_trace = {
            "raw_memory_count": len(raw_memories),
            "memory_count": len(self.store.items),
            "writer_trace": self.writer.last_trace,
        }

    def answer(self, question: str) -> str:
        retrieved = self.retriever.retrieve(question)
        memory_lines = [
            (
                f"- [{result.item.timestamp}] {result.item.text} "
                f"(source={result.item.source}, score={result.score:.3f}, "
                f"rel={result.relevance:.3f}, rec={result.recency:.3f}, "
                f"imp={result.importance:.3f})"
            )
            for result in retrieved
        ]
        prompt = (
            "You are a long-term memory dialogue agent. Answer the question using only "
            "the memory units below. Keep the answer short. If the answer is absent, "
            "reply 'unknown'.\n\n"
            f"=== Memory units ===\n{chr(10).join(memory_lines)}\n\n"
            f"=== Question ===\n{question}\n\n"
            "=== Answer ==="
        )
        answer = self.llm.tracked_generate(prompt, max_tokens=64).strip()
        self.last_trace = {
            "memory_count": len(self.store.items),
            "retrieved": [
                {
                    "text": result.item.text,
                    "source": result.item.source,
                    "timestamp": result.item.timestamp,
                    "importance": result.item.importance,
                    "score": round(result.score, 4),
                    "relevance": round(result.relevance, 4),
                    "recency": round(result.recency, 4),
                    "importance_score": round(result.importance, 4),
                    "metadata": result.item.metadata,
                }
                for result in retrieved
            ],
            "prompt": prompt,
            "llm": self.llm.snapshot(),
        }
        return answer

    def get_trace(self) -> dict:
        return self.last_trace
