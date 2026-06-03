from dataclasses import dataclass

from memory_agent.memory.store import MemoryItem, MemoryStore


@dataclass
class RetrievalResult:
    item: MemoryItem
    score: float
    relevance: float
    recency: float
    importance: float


class MemoryRetriever:
    """Generative-Agents style retrieval: relevance, recency, and importance."""

    def __init__(
        self,
        store: MemoryStore,
        top_k: int = 8,
        relevance_weight: float = 0.70,
        recency_weight: float = 0.15,
        importance_weight: float = 0.15,
    ):
        self.store = store
        self.top_k = top_k
        self.relevance_weight = relevance_weight
        self.recency_weight = recency_weight
        self.importance_weight = importance_weight

    def retrieve(self, question: str) -> list[RetrievalResult]:
        relevance_scores = self.store.relevance_scores(question)
        if not relevance_scores:
            return []
        results = []
        for item, relevance in zip(self.store.items, relevance_scores):
            recency = self.store.recency_score(item)
            importance = min(max(item.importance / 5.0, 0.0), 1.0)
            score = (
                self.relevance_weight * relevance
                + self.recency_weight * recency
                + self.importance_weight * importance
            )
            results.append(RetrievalResult(item, score, relevance, recency, importance))
        results.sort(key=lambda x: x.score, reverse=True)
        selected = results[: self.top_k]
        for result in selected:
            self.store.mark_accessed(result.item)
        return selected

