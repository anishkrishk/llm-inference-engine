"""
A trivial next-token model for exercising the scheduler without a real LLM.

The dummy returns one pseudorandom token per running sequence per step.
Token ids are a deterministic function of (request_id, current_output_len)
rather than drawn from an RNG. That property gives us two things for free:

  1. Tests can assert exact token streams without mocking a time-dependent
     RNG.
  2. A sequence that is preempted and later recomputed produces the same
     token stream as one that ran straight through — exactly the
     correctness invariant a real recompute-based preemption path has to
     uphold. We can therefore test preemption end-to-end without a real
     attention implementation.

The model is intentionally ignorant of shapes, dtypes, and CUDA. The
scheduler is what we're testing here, not the math.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Dict

from src.scheduler.scheduler import SchedulerOutput


@dataclass
class DummyModelConfig:
    vocab_size: int = 50257  # GPT-2 vocab size, kept for realism


class DummyModel:
    def __init__(self, config: DummyModelConfig) -> None:
        self._config = config

    def step(self, output: SchedulerOutput) -> Dict[str, int]:
        """
        Produce one next-token id per running sequence. Prefill sequences
        and decode sequences are treated identically: both get one token
        this step. In a real model the prefill pass would compute KV for
        the full prompt; here we only care that the scheduler gets a
        token back for every sequence it asked about.
        """
        tokens: Dict[str, int] = {}
        for seq in output.iter_seqs():
            # blake2b gives us a cheap stable hash with no RNG state. The
            # key is (request_id, output_len) so two sequences in flight
            # with different histories don't collide, and so the k-th
            # token of a given request is always the same value.
            digest = hashlib.blake2b(
                f"{seq.request_id}:{seq.output_len}".encode("utf-8"),
                digest_size=8,
            ).digest()
            tokens[seq.request_id] = int.from_bytes(digest, "big") % self._config.vocab_size
        return tokens
