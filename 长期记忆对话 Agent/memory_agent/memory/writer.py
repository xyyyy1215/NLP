import json
import re

from memory_agent.eval.core.llm_client import LLMClient
from memory_agent.memory.store import MemoryItem


WRITE_SYSTEM = (
    "You extract high-fidelity long-term memories from dialogue. "
    "Return JSON only. Preserve concrete facts; do not over-summarize."
)

WRITE_PROMPT = """Extract long-term memory facts from this dialogue session.

Keep only stable or useful information for future questions:
- identity, relationship, preference, plan, event, location, job, family, health, update, or conflict.
- names, dates, relative time phrases, activities, hobbies, objects, books, events, and opinions.
- Keep concrete answerable details even if they are one-off events.
- Preserve exact wording for dates and relative times such as yesterday, last Friday, this month, or next month.
- Split different facts into separate atomic memory items.
- Ignore greetings, compliments, and generic emotional support unless they reveal an opinion or relationship.

Return a JSON array. Each item must have:
{{
  "text": "one short standalone memory sentence",
  "subject": "person or entity",
  "attribute": "preference/location/job/event/relationship/plan/update/other",
  "importance": 1-5
}}

Session time: {timestamp}
Dialogue:
{dialogue}
"""


class MemoryWriter:
    """LLM-based memory extraction inspired by Mem0's write stage."""

    def __init__(self, llm: LLMClient, max_turns_per_call: int = 24):
        self.llm = llm
        self.max_turns_per_call = max_turns_per_call
        self.last_trace: list[dict] = []

    def write(self, conversation: dict) -> list[MemoryItem]:
        memories: list[MemoryItem] = []
        self.last_trace = []
        for session in conversation.get("sessions", []):
            chunks = self._session_chunks(session)
            for chunk_index, dialogue in enumerate(chunks):
                prompt = WRITE_PROMPT.format(
                    timestamp=session.get("date_time", ""),
                    dialogue=dialogue,
                )
                raw = self.llm.tracked_generate(
                    prompt,
                    max_tokens=768,
                    temperature=0.0,
                    system=WRITE_SYSTEM,
                )
                parsed = self._parse(raw)
                if not parsed:
                    parsed = self._fallback(dialogue)
                source = f"{session.get('session_id', '')}:chunk{chunk_index}"
                for obj in parsed:
                    text = str(obj.get("text", "")).strip()
                    if not text:
                        continue
                    memories.append(
                        MemoryItem(
                            text=text,
                            source=source,
                            timestamp=session.get("date_time", ""),
                            importance=_importance(obj.get("importance", 2)),
                            metadata={
                                "subject": str(obj.get("subject", "")).strip().lower(),
                                "attribute": str(obj.get("attribute", "other")).strip().lower(),
                                "writer": "llm",
                            },
                        )
                    )
                self.last_trace.append(
                    {
                        "session_id": session.get("session_id", ""),
                        "chunk_index": chunk_index,
                        "raw": raw,
                        "parsed_count": len(parsed),
                    }
                )
        return memories

    def _session_chunks(self, session: dict) -> list[str]:
        lines = [
            f"{turn.get('speaker', 'unknown')}: {turn.get('text', '')}"
            for turn in session.get("turns", [])
            if str(turn.get("text", "")).strip()
        ]
        chunks = []
        for start in range(0, len(lines), self.max_turns_per_call):
            chunks.append("\n".join(lines[start : start + self.max_turns_per_call]))
        return chunks or [""]

    def _parse(self, raw: str) -> list[dict]:
        if not raw:
            return []
        text = re.sub(r"^```(?:json)?\s*", "", raw.strip())
        text = re.sub(r"\s*```$", "", text)
        try:
            obj = json.loads(text)
        except json.JSONDecodeError:
            match = re.search(r"\[[\s\S]*\]", text)
            if not match:
                return []
            try:
                obj = json.loads(match.group(0))
            except json.JSONDecodeError:
                return []
        if isinstance(obj, dict):
            obj = obj.get("memories", [])
        return [x for x in obj if isinstance(x, dict)]

    def _fallback(self, dialogue: str) -> list[dict]:
        memories = []
        cues = (
            "adopt",
            "birthday",
            "book",
            "camp",
            "conference",
            "favorite",
            "family",
            "job",
            "last",
            "lgbt",
            "lives",
            "moved",
            "plan",
            "prefer",
            "research",
            "transgender",
            "works",
            "yesterday",
        )
        for line in dialogue.splitlines():
            lowered = line.lower()
            if any(cue in lowered for cue in cues):
                memories.append(
                    {
                        "text": line.strip(),
                        "subject": line.split(":", 1)[0].strip() if ":" in line else "",
                        "attribute": "other",
                        "importance": 2,
                    }
                )
        return memories[:6]


def _importance(value) -> float:
    try:
        return max(1.0, min(float(value), 5.0))
    except (TypeError, ValueError):
        return 2.0
