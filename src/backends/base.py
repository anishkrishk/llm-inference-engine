from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, List, Optional


@dataclass
class GenerateInput:
    request_id: str
    prompt: str
    max_new_tokens: int
    temperature: float
    top_p: float
    seed: int
    stop_sequences: List[str]
    model_id: str


@dataclass
class TokenEvent:
    token_id: int
    token_text: str
    token_index: int


@dataclass
class GenerateResult:
    text: str
    token_ids: List[int]
    prompt_tokens: int
    generated_tokens: int
    finish_reason: str


class Backend:
    def generate(self, req: GenerateInput) -> GenerateResult:
        raise NotImplementedError

    def generate_stream(self, req: GenerateInput) -> Iterator[TokenEvent]:
        raise NotImplementedError

    def finalize_stream(self, req: GenerateInput, events: List[TokenEvent]) -> GenerateResult:
        raise NotImplementedError