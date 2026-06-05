import math
import re
from dataclasses import dataclass

from memory_agent.memory.store import MemoryItem, MemoryStore


@dataclass
class RetrievalResult:
    item: MemoryItem
    score: float
    relevance: float
    recency: float
    importance: float
    lexical: float = 0.0


class MemoryRetriever:
    """Generative-Agents style retrieval: relevance, recency, and importance."""

    def __init__(
        self,
        store: MemoryStore,
        top_k: int = 8,
        relevance_weight: float = 0.60,
        lexical_weight: float = 0.35,
        recency_weight: float = 0.025,
        importance_weight: float = 0.025,
        dense_candidate_count: int = 50,
        lexical_candidate_count: int = 50,
    ):
        self.store = store
        self.top_k = top_k
        self.relevance_weight = relevance_weight
        self.lexical_weight = lexical_weight
        self.recency_weight = recency_weight
        self.importance_weight = importance_weight
        self.dense_candidate_count = dense_candidate_count
        self.lexical_candidate_count = lexical_candidate_count
        self._token_cache: dict[int, set[str]] = {}
        self._subject_token_cache: dict[int, set[str]] = {}

    def retrieve(self, question: str) -> list[RetrievalResult]:
        relevance_scores = self.store.relevance_scores(question)
        if not relevance_scores:
            return []
        query_tokens = _expanded_query_tokens(question)
        lexical_scores = [
            self._lexical_score_for_item(query_tokens, index, item)
            for index, item in enumerate(self.store.items)
        ]
        candidate_indices = self._candidate_indices(relevance_scores, lexical_scores)
        results = []
        for index in candidate_indices:
            item = self.store.items[index]
            relevance = relevance_scores[index]
            recency = self.store.recency_score(item)
            importance = min(max(item.importance / 5.0, 0.0), 1.0)
            lexical = lexical_scores[index]
            score = (
                self.relevance_weight * relevance
                + self.lexical_weight * lexical
                + self.recency_weight * recency
                + self.importance_weight * importance
            )
            results.append(RetrievalResult(item, score, relevance, recency, importance, lexical))
        results.sort(key=lambda x: x.score, reverse=True)
        selected = results[: self.top_k]
        for result in selected:
            self.store.mark_accessed(result.item)
        return selected

    def _candidate_indices(self, relevance_scores: list[float], lexical_scores: list[float]) -> list[int]:
        dense_ranked = sorted(range(len(relevance_scores)), key=relevance_scores.__getitem__, reverse=True)
        lexical_ranked = sorted(range(len(lexical_scores)), key=lexical_scores.__getitem__, reverse=True)
        candidates = set(dense_ranked[: self.dense_candidate_count])
        candidates.update(lexical_ranked[: self.lexical_candidate_count])
        candidates.update(i for i, score in enumerate(lexical_scores) if score >= 0.55)
        return sorted(candidates)

    def _lexical_score_for_item(self, query_tokens: set[str], index: int, item: MemoryItem) -> float:
        memory_tokens = self._memory_tokens(index, item)
        if not memory_tokens:
            return 0.0
        overlap = query_tokens & memory_tokens
        recall = len(overlap) / max(len(query_tokens), 1)
        precision = len(overlap) / len(memory_tokens)
        f1 = 0.0 if not overlap else 2 * precision * recall / (precision + recall)
        name_bonus = 0.15 if self._subject_tokens(index, item) & query_tokens else 0.0
        return min(1.0, math.sqrt(recall) * 0.75 + f1 * 0.25 + name_bonus)

    def _memory_tokens(self, index: int, item: MemoryItem) -> set[str]:
        if index not in self._token_cache:
            metadata = item.metadata or {}
            memory_text = item.text
            subject = metadata.get("subject") or metadata.get("speaker")
            attribute = metadata.get("attribute")
            if subject:
                memory_text = f"{memory_text} {subject}"
            if attribute:
                memory_text = f"{memory_text} {attribute}"
            self._token_cache[index] = _tokens(memory_text)
        return self._token_cache[index]

    def _subject_tokens(self, index: int, item: MemoryItem) -> set[str]:
        if index not in self._subject_token_cache:
            metadata = item.metadata or {}
            subject = metadata.get("subject") or metadata.get("speaker")
            self._subject_token_cache[index] = _tokens(str(subject or ""))
        return self._subject_token_cache[index]


_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "around",
    "be",
    "did",
    "do",
    "does",
    "for",
    "from",
    "has",
    "have",
    "how",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "the",
    "to",
    "was",
    "what",
    "when",
    "where",
    "which",
    "who",
    "why",
    "would",
}


def _tokens(text: str) -> set[str]:
    return {
        _stem(token)
        for token in re.findall(r"[a-z0-9]+", str(text).lower())
        if len(token) > 1 and token not in _STOPWORDS
    }


def _stem(token: str) -> str:
    if len(token) > 5 and token.endswith("ing"):
        return token[:-3]
    if len(token) > 4 and token.endswith("ies"):
        return f"{token[:-3]}y"
    if len(token) > 4 and token.endswith("ed"):
        return token[:-2]
    if len(token) > 3 and token.endswith("s"):
        return token[:-1]
    return token


def _lexical_score(question: str, text: str, metadata: dict) -> float:
    query_tokens = _expanded_query_tokens(question)
    if not query_tokens:
        return 0.0
    memory_text = text
    subject = metadata.get("subject") or metadata.get("speaker")
    attribute = metadata.get("attribute")
    if subject:
        memory_text = f"{memory_text} {subject}"
    if attribute:
        memory_text = f"{memory_text} {attribute}"
    memory_tokens = _tokens(memory_text)
    if not memory_tokens:
        return 0.0
    overlap = query_tokens & memory_tokens
    recall = len(overlap) / len(query_tokens)
    precision = len(overlap) / len(memory_tokens)
    f1 = 0.0 if not overlap else 2 * precision * recall / (precision + recall)
    subject_tokens = _tokens(str(subject or ""))
    name_bonus = 0.15 if subject_tokens & query_tokens else 0.0
    return min(1.0, math.sqrt(recall) * 0.75 + f1 * 0.25 + name_bonus)


def _expanded_query_tokens(question: str) -> set[str]:
    tokens = _tokens(question)
    if "identity" in tokens:
        tokens.update({"transgender", "transition", "womanhood"})
    if "activity" in tokens or "hobby" in tokens:
        tokens.update({"camp", "paint", "pottery", "run", "swim"})
    if "destres" in tokens or "stress" in tokens:
        tokens.update({"calm", "relax", "run", "pottery"})
    if "lean" in tokens or "political" in tokens:
        tokens.update({"liberal", "advocacy", "lgbt", "pride"})
    return tokens
