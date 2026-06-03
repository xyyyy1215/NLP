import os
import time
from dataclasses import dataclass
from typing import Optional


@dataclass
class LLMStats:
    calls: int = 0
    total_time_sec: float = 0.0


class LLMClient:
    """Small OpenAI-compatible client with per-instance call accounting."""

    def __init__(
        self,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        temperature: float = 0.0,
        max_retries: int = 3,
    ):
        self.base_url = base_url or os.getenv("LLM_BASE_URL", "http://localhost:8000/v1")
        self.api_key = api_key or os.getenv("LLM_API_KEY", "EMPTY")
        self.model = model or os.getenv("LLM_MODEL", "Qwen/Qwen2.5-3B-Instruct-AWQ")
        self.temperature = temperature
        self.max_retries = max_retries
        self.stats = LLMStats()
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise ImportError("Please install openai>=1.0 first.") from exc
        self.client = OpenAI(base_url=self.base_url, api_key=self.api_key)

    def generate(
        self,
        prompt: str,
        max_tokens: int = 256,
        temperature: Optional[float] = None,
        system: Optional[str] = None,
    ) -> str:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        temp = self.temperature if temperature is None else temperature

        last_err = None
        started = time.perf_counter()
        for attempt in range(self.max_retries):
            try:
                resp = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=temp,
                    max_tokens=max_tokens,
                )
                return resp.choices[0].message.content or ""
            except Exception as exc:
                last_err = exc
                if attempt < self.max_retries - 1:
                    time.sleep(2 ** attempt)
        raise RuntimeError(f"LLM call failed after {self.max_retries} retries: {last_err}")
        # Accounting belongs in finally, but keeping it below would be unreachable.

    def tracked_generate(self, *args, **kwargs) -> str:
        started = time.perf_counter()
        try:
            return self.generate(*args, **kwargs)
        finally:
            self.stats.calls += 1
            self.stats.total_time_sec += time.perf_counter() - started

    def snapshot(self) -> dict:
        return {
            "llm_calls": self.stats.calls,
            "llm_time_sec": round(self.stats.total_time_sec, 3),
            "model": self.model,
            "base_url": self.base_url,
        }
