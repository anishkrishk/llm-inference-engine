"""
Plain-data types passed between the gRPC service layer and the inference core.

These dataclasses are deliberately framework-agnostic: no torch tensors, no
protobuf objects. The service layer translates protobuf <-> these dataclasses,
and the inference core consumes/produces them. Keeping the boundary narrow
means we can swap either side without touching the other.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List


@dataclass
class GenerateInput:
    """A single generation request as it crosses into the inference core."""

    request_id: str
    prompt: str
    max_new_tokens: int
    temperature: float
    top_p: float
    seed: int
    stop_sequences: List[str] = field(default_factory=list)
    model_id: str = ""


@dataclass
class TokenEvent:
    """One generated token, emitted incrementally during streaming."""

    token_id: int
    token_text: str
    token_index: int


@dataclass
class GenerateResult:
    """Terminal result of a completed generation."""

    text: str
    token_ids: List[int]
    prompt_tokens: int
    generated_tokens: int
    finish_reason: str
