import re

from memory_agent.memory.store import MemoryItem


class MemoryUpdater:
    """Deduplicate memories and keep newer facts for the same subject/attribute."""

    def merge(self, memories: list[MemoryItem]) -> list[MemoryItem]:
        by_text: dict[str, MemoryItem] = {}
        by_slot: dict[tuple[str, str], MemoryItem] = {}
        for memory in memories:
            text_key = _normalize(memory.text)
            if text_key in by_text:
                existing = by_text[text_key]
                existing.importance = max(existing.importance, memory.importance)
                continue

            slot = self._slot(memory)
            if slot is not None and slot in by_slot:
                old = by_slot[slot]
                if _is_newer(memory, old):
                    old.metadata["updated_by"] = memory.source
                    by_slot[slot] = memory
                    by_text.pop(_normalize(old.text), None)
                    by_text[text_key] = memory
                continue

            by_text[text_key] = memory
            if slot is not None:
                by_slot[slot] = memory
        return list(by_text.values())

    def _slot(self, memory: MemoryItem) -> tuple[str, str] | None:
        subject = str(memory.metadata.get("subject", "")).strip().lower()
        attribute = str(memory.metadata.get("attribute", "")).strip().lower()
        if not subject or attribute in {"", "event", "other"}:
            return None
        return subject, attribute


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower()).strip()


def _is_newer(left: MemoryItem, right: MemoryItem) -> bool:
    if left.timestamp != right.timestamp:
        return str(left.timestamp) > str(right.timestamp)
    return left.source >= right.source

