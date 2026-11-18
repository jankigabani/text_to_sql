from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol


@dataclass
class LLMResult:
    text: str
    model: str
    prompt_chars: int = 0
    completion_chars: int = 0
    latency_ms: float = 0.0
    meta: dict = field(default_factory=dict)

    @property
    def prompt_tokens_est(self) -> int:
        return self.prompt_chars // 4


class LLMProvider(Protocol):
    name: str
    model: str

    def complete(self, system: str, user: str) -> LLMResult: ...

    def health(self) -> tuple[bool, str]: ...
