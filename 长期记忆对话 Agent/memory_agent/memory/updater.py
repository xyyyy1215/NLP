import re

from memory_agent.memory.store import MemoryItem


class MemoryUpdater:
    """Deduplicate memories without dropping distinct facts for the same person."""

    def merge(self, memories: list[MemoryItem]) -> list[MemoryItem]:
        by_text: dict[str, MemoryItem] = {}
        for memory in memories:
            text_key = _normalize(memory.text)
            if text_key in by_text:
                existing = by_text[text_key]
                existing.importance = max(existing.importance, memory.importance)
                continue

            by_text[text_key] = memory
        return list(by_text.values())


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower()).strip()
