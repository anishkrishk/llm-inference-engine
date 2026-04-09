"""
Minimal inference engine wiring together the scheduler, block allocator,
and a pluggable next-token model.

The engine owns the policy-free plumbing:

  * accept generation requests (with pre-tokenized prompts)
  * drive the scheduler's step loop
  * hand SchedulerOutput to the model, hand the model's output back to
    the scheduler for commit
  * collect finished sequences so callers can retrieve results

Keeping the engine this small means swapping the scheduler policy or
the model implementation touches one object at a time. It also means
tests can instantiate an engine in a couple of lines.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Protocol

from src.kv_cache.allocator import BlockAllocator
from src.kv_cache.block_table import BlockTable
from src.model.types import GenerateInput
from src.scheduler.scheduler import Scheduler, SchedulerConfig, SchedulerOutput
from src.scheduler.sequence import Sequence


class ModelLike(Protocol):
    def step(self, output: SchedulerOutput) -> Dict[str, int]: ...


@dataclass
class EngineConfig:
    num_blocks: int
    block_size: int = 16
    max_num_seqs: int = 16
    max_num_batched_tokens: int = 4096


class Engine:
    def __init__(self, config: EngineConfig, model: ModelLike) -> None:
        self._config = config
        self._allocator = BlockAllocator(config.num_blocks)
        self._scheduler = Scheduler(
            self._allocator,
            SchedulerConfig(
                block_size=config.block_size,
                max_num_seqs=config.max_num_seqs,
                max_num_batched_tokens=config.max_num_batched_tokens,
            ),
        )
        self._model = model
        self._results: Dict[str, Sequence] = {}

    # ------------------------------------------------------------------
    # Request lifecycle
    # ------------------------------------------------------------------

    def add_request(
        self, req: GenerateInput, prompt_token_ids: List[int]
    ) -> None:
        """
        Submit a new request to the engine. The caller provides the
        already-tokenized prompt; the engine does not own a tokenizer.
        """
        table = BlockTable(self._allocator, self._config.block_size)
        seq = Sequence(
            request_id=req.request_id,
            prompt_token_ids=list(prompt_token_ids),
            max_new_tokens=req.max_new_tokens,
            block_table=table,
        )
        self._scheduler.add_sequence(seq)

    def step(self) -> SchedulerOutput:
        """Advance one scheduling iteration. Runs the model if there is work."""
        output = self._scheduler.step()
        if output.is_empty:
            return output
        next_tokens = self._model.step(output)
        finished = self._scheduler.commit(output, next_tokens)
        for seq in finished:
            self._results[seq.request_id] = seq
        return output

    def run_until_done(self, max_steps: int = 10_000) -> int:
        """
        Drive step() until no work remains. Returns the number of steps
        taken. `max_steps` is a guard against accidental infinite loops
        under pathological configurations.
        """
        steps = 0
        while self.has_work() and steps < max_steps:
            self.step()
            steps += 1
        if self.has_work():
            raise RuntimeError(
                f"engine did not finish within {max_steps} steps; "
                f"remaining: running={self._scheduler.num_running}, "
                f"waiting={self._scheduler.num_waiting}"
            )
        return steps

    def has_work(self) -> bool:
        return self._scheduler.has_work

    def get_result(self, request_id: str) -> Optional[Sequence]:
        return self._results.get(request_id)

    # ------------------------------------------------------------------
    # Observability
    # ------------------------------------------------------------------

    @property
    def num_free_blocks(self) -> int:
        return self._allocator.num_free

    @property
    def num_total_blocks(self) -> int:
        return self._allocator.num_total

    @property
    def num_running(self) -> int:
        return self._scheduler.num_running

    @property
    def num_waiting(self) -> int:
        return self._scheduler.num_waiting
