"""
Per-request state carried through the scheduler.

A Sequence is the unit of scheduling: it owns its prompt token ids, its
generated-so-far output tokens, a BlockTable describing where its KV
cache lives in the physical pool, and a status enum identifying which
queue it currently belongs to.

Notably, the Sequence does NOT hold any tensors. All "cached" KV lives
in the shared pool identified by the block ids in its BlockTable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import List

from src.kv_cache.block_table import BlockTable


class SequenceStatus(Enum):
    WAITING = "waiting"            # queued; BlockTable has no blocks reserved
    RUNNING = "running"            # admitted into the active set; KV is populated
    FINISHED_STOPPED = "stopped"   # hit a stop-token / stop-sequence condition
    FINISHED_LENGTH = "length"     # hit max_new_tokens
    FINISHED_ABORTED = "aborted"   # cancelled by the client

    @classmethod
    def finished_states(cls) -> frozenset["SequenceStatus"]:
        return frozenset(
            {cls.FINISHED_STOPPED, cls.FINISHED_LENGTH, cls.FINISHED_ABORTED}
        )


@dataclass
class Sequence:
    request_id: str
    prompt_token_ids: List[int]
    max_new_tokens: int
    block_table: BlockTable
    status: SequenceStatus = SequenceStatus.WAITING
    output_token_ids: List[int] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.max_new_tokens < 1:
            raise ValueError(
                f"max_new_tokens must be >= 1, got {self.max_new_tokens}"
            )
        if not self.prompt_token_ids:
            raise ValueError("prompt_token_ids must not be empty")

    @property
    def prompt_len(self) -> int:
        return len(self.prompt_token_ids)

    @property
    def output_len(self) -> int:
        return len(self.output_token_ids)

    @property
    def total_len(self) -> int:
        return self.prompt_len + self.output_len

    def all_token_ids(self) -> List[int]:
        return self.prompt_token_ids + self.output_token_ids

    def append_output(self, token_id: int) -> None:
        self.output_token_ids.append(token_id)

    def is_finished(self) -> bool:
        return self.status in SequenceStatus.finished_states()

    def __repr__(self) -> str:
        return (
            f"Sequence(id={self.request_id!r}, status={self.status.value}, "
            f"prompt_len={self.prompt_len}, output_len={self.output_len}, "
            f"num_blocks={self.block_table.num_blocks})"
        )
