import math
import os
from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class MemoryItem:
    text: str
    source: str
    timestamp: str = ""
    importance: float = 1.0
    metadata: dict = field(default_factory=dict)
    created_index: int = 0
    access_count: int = 0


class MemoryStore:
    """Memory storage with a local sentence-transformer embedding index."""

    def __init__(self, embed_model: str | None = None):
        self.items: list[MemoryItem] = []
        self.embeddings = None
        self.embed_model_name = embed_model or os.getenv("EMBED_MODEL", "models/bge-m3")
        self.embed_device = os.getenv("EMBED_DEVICE", "cpu")
        self._embed_model = None

    def add(self, item: MemoryItem) -> None:
        item.created_index = len(self.items)
        self.items.append(item)

    def replace(self, index: int, item: MemoryItem) -> None:
        item.created_index = self.items[index].created_index
        self.items[index] = item

    def rebuild(self) -> None:
        if not self.items:
            self.embeddings = None
            return
        import numpy as np

        model = self._model()
        texts = [_index_text(item) for item in self.items]
        vecs = model.encode(texts, normalize_embeddings=True)
        self.embeddings = np.array(vecs, dtype=np.float32)

    def relevance_scores(self, query: str) -> list[float]:
        if not self.items:
            return []
        if self.embeddings is None or len(self.embeddings) != len(self.items):
            self.rebuild()
        import numpy as np

        qvec = self._model().encode([query], normalize_embeddings=True)[0]
        sims = self.embeddings @ qvec.astype(np.float32)
        return [float(x) for x in sims]

    def mark_accessed(self, item: MemoryItem) -> None:
        item.access_count += 1

    def recency_score(self, item: MemoryItem) -> float:
        parsed = _parse_time(item.timestamp)
        if parsed is None:
            # Fall back to insertion order when timestamps are not parseable.
            return 1.0 / (1.0 + max(len(self.items) - item.created_index - 1, 0))
        latest = max((_parse_time(x.timestamp) for x in self.items), default=None)
        if latest is None:
            return 1.0
        days = max((latest - parsed).days, 0)
        return math.exp(-days / 30.0)

    def _model(self):
        if self._embed_model is None:
            from sentence_transformers import SentenceTransformer

            self._embed_model = SentenceTransformer(self.embed_model_name, device=self.embed_device)
        return self._embed_model


def _parse_time(value: str) -> datetime | None:
    if not value:
        return None
    value = str(value).strip()
    formats = (
        ("%Y-%m-%d %H:%M:%S", 19),
        ("%Y-%m-%d", 10),
        ("%m/%d/%Y", 10),
        ("%d/%m/%Y", 10),
    )
    for fmt, width in formats:
        try:
            return datetime.strptime(value[:width], fmt)
        except ValueError:
            continue
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).replace(tzinfo=None)
    except ValueError:
        return None


def _index_text(item: MemoryItem) -> str:
    metadata = item.metadata or {}
    parts = [item.text]
    subject = metadata.get("subject") or metadata.get("speaker")
    attribute = metadata.get("attribute")
    if subject:
        parts.append(f"subject: {subject}")
    if attribute:
        parts.append(f"type: {attribute}")
    if item.timestamp:
        parts.append(f"time: {item.timestamp}")
    return " ".join(str(part) for part in parts if str(part).strip())
