import calendar
import os
import re
from datetime import timedelta

from memory_agent.eval.core.llm_client import LLMClient
from memory_agent.memory.retriever import MemoryRetriever
from memory_agent.memory.store import MemoryItem, MemoryStore, _parse_time
from memory_agent.memory.updater import MemoryUpdater
from memory_agent.memory.writer import MemoryWriter


class MemoryAgent:
    """Main long-term memory agent: write memories, retrieve them, then answer."""

    def __init__(self, top_k: int = 8):
        self.llm = LLMClient()
        self.store = MemoryStore()
        self.use_writer = _env_bool("MEMORY_AGENT_USE_WRITER", default=False)
        self.use_raw_turns = _env_bool("MEMORY_AGENT_USE_RAW_TURNS", default=True)
        self.use_temporal_hints = _env_bool("MEMORY_AGENT_USE_TEMPORAL_HINT", default=True)
        top_k = int(os.getenv("MEMORY_AGENT_TOP_K", str(top_k)))
        self.writer = MemoryWriter(self.llm) if self.use_writer else None
        self.updater = MemoryUpdater()
        self.retriever = MemoryRetriever(self.store, top_k=top_k)
        self.last_trace: dict = {}

    def ingest(self, conversation: dict) -> None:
        raw_memories = []
        writer_trace = []
        if self.writer is not None:
            raw_memories = self.writer.write(conversation)
            memories = self.updater.merge(raw_memories)
            for memory in memories:
                self.store.add(memory)
            writer_trace = self.writer.last_trace
        raw_turns = []
        if self.use_raw_turns:
            raw_turns = self._raw_turn_memories(conversation)
            for memory in raw_turns:
                self.store.add(memory)
        self.store.rebuild()
        self.last_trace = {
            "use_writer": self.use_writer,
            "use_raw_turns": self.use_raw_turns,
            "use_temporal_hints": self.use_temporal_hints,
            "raw_memory_count": len(raw_memories),
            "raw_turn_count": len(raw_turns),
            "memory_count": len(self.store.items),
            "writer_trace": writer_trace,
        }

    def answer(self, question: str) -> str:
        retrieved = self.retriever.retrieve(question)
        memory_lines = [
            _evidence_line(result.item)
            for result in retrieved
        ]
        prompt = (
            "You are a long-term memory dialogue agent. Answer using only the memory "
            "units and raw dialogue evidence below. Return a short answer phrase, not "
            "an explanation. If the evidence contains a relative time such as yesterday, "
            "last Friday, this month, or next month, infer the date from the timestamp "
            "shown in brackets. If several evidence lines are relevant, combine them. "
            "Reply 'unknown' only when no relevant evidence is present.\n\n"
            f"=== Memory units ===\n{chr(10).join(memory_lines)}\n\n"
            f"=== Question ===\n{question}\n\n"
            "=== Answer ==="
        )
        answer = self.llm.tracked_generate(prompt, max_tokens=96).strip()
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
                    "lexical": round(result.lexical, 4),
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

    def _raw_turn_memories(self, conversation: dict) -> list[MemoryItem]:
        memories: list[MemoryItem] = []
        for session in conversation.get("sessions", []):
            timestamp = session.get("date_time", "")
            session_id = session.get("session_id", "")
            for turn_index, turn in enumerate(session.get("turns", [])):
                text = str(turn.get("text", "")).strip()
                if not text:
                    continue
                speaker = str(turn.get("speaker", "unknown")).strip()
                evidence_text = f"[{timestamp}] {speaker}: {text}"
                time_hints = _temporal_hints(text, timestamp) if self.use_temporal_hints else []
                if time_hints:
                    evidence_text = f"{evidence_text} Inferred time hints: {'; '.join(time_hints)}."
                memories.append(
                    MemoryItem(
                        text=evidence_text,
                        source=f"{session_id}:turn{turn_index}",
                        timestamp=timestamp,
                        importance=1.5,
                        metadata={
                            "kind": "raw_turn",
                            "speaker": speaker.lower(),
                            "session_id": session_id,
                        },
                    )
                )
        return memories


_WEEKDAYS = {
    "monday": ("monday", 0),
    "mon": ("monday", 0),
    "tuesday": ("tuesday", 1),
    "tue": ("tuesday", 1),
    "tues": ("tuesday", 1),
    "wednesday": ("wednesday", 2),
    "wed": ("wednesday", 2),
    "thursday": ("thursday", 3),
    "thu": ("thursday", 3),
    "thur": ("thursday", 3),
    "thurs": ("thursday", 3),
    "friday": ("friday", 4),
    "fri": ("friday", 4),
    "saturday": ("saturday", 5),
    "sat": ("saturday", 5),
    "sunday": ("sunday", 6),
    "sun": ("sunday", 6),
}


def _temporal_hints(text: str, timestamp: str) -> list[str]:
    base = _parse_time(timestamp)
    if base is None:
        return []
    lowered = str(text).lower()
    hints: list[str] = []

    month_year = _month_year(base.month, base.year)
    previous_month = _shift_month(base.month, base.year, -1)
    next_month = _shift_month(base.month, base.year, 1)
    phrases = [
        ("this month", month_year),
        ("last month", _month_year(*previous_month)),
        ("next month", _month_year(*next_month)),
        ("last year", str(base.year - 1)),
        ("this year", str(base.year)),
        ("next year", str(base.year + 1)),
        ("yesterday", _date_label(base - timedelta(days=1))),
        ("today", _date_label(base)),
        ("tomorrow", _date_label(base + timedelta(days=1))),
        ("last week", f"the week before {_date_label(base)}"),
        ("next week", f"the week after {_date_label(base)}"),
    ]
    for phrase, value in phrases:
        if phrase in lowered:
            hints.append(f"{phrase} = {value}")

    weekday_pattern = "|".join(sorted(_WEEKDAYS, key=len, reverse=True))
    for match in re.finditer(rf"\b(last|next)\s+({weekday_pattern})\b", lowered):
        direction, weekday_key = match.groups()
        weekday_name, weekday = _WEEKDAYS[weekday_key]
        delta = (base.weekday() - weekday) % 7 if direction == "last" else (weekday - base.weekday()) % 7
        if delta == 0:
            delta = 7
        resolved = base - timedelta(days=delta) if direction == "last" else base + timedelta(days=delta)
        hints.append(f"{direction} {weekday_name} = {_date_label(resolved)}")

    return _dedupe(hints)


def _shift_month(month: int, year: int, offset: int) -> tuple[int, int]:
    zero_based = month - 1 + offset
    new_year = year + zero_based // 12
    new_month = zero_based % 12 + 1
    return new_month, new_year


def _month_year(month: int, year: int) -> str:
    return f"{calendar.month_name[month]} {year}"


def _date_label(value) -> str:
    return f"{value.day} {calendar.month_name[value.month]} {value.year}"


def _dedupe(items: list[str]) -> list[str]:
    seen = set()
    output = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        output.append(item)
    return output


def _evidence_line(item: MemoryItem) -> str:
    text = str(item.text).strip()
    timestamp = str(item.timestamp).strip()
    if timestamp and not text.startswith("["):
        text = f"[{timestamp}] {text}"
    return f"- {text}"


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}
