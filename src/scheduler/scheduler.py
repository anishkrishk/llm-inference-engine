"""
Iteration-level (continuous) scheduler over a shared KV block pool.

Responsibilities
----------------
The scheduler holds the three queues that every iteration-level batching
engine needs:

    waiting   :  sequences that have never run, OR were preempted and need
                 to re-prefill their full history
    running   :  sequences currently consuming KV in the pool
    finished  :  terminal sequences whose blocks have already been freed

Each call to step() does exactly this:

    1. Advance every running sequence's KV-cache pointer by one token
       (the one produced by the previous step). If this requires a new
       physical block and the pool is full, preempt a sequence from the
       running set (LIFO) and try again.

    2. Drain the waiting queue into running for as long as (a) the
       concurrency cap allows, (b) the total-tokens budget allows, and
       (c) the allocator can satisfy the prefill.

    Then return a SchedulerOutput carrying
        - prefill_seqs : just-admitted seqs (need a full prompt pass)
        - decode_seqs  : already-running seqs (need one-token step)
        - preempted_seqs : seqs evicted in step (1), for observability

After the caller has run the model over the SchedulerOutput and has a
{request_id -> next_token} map, it calls commit() to append the newly
produced tokens, mark finished sequences, and free their blocks.

Invariant the scheduler maintains
---------------------------------
For every running sequence AT THE START OF A STEP:

    seq.block_table.num_tokens == seq.total_len - 1    (mid-decode)
    seq.block_table.num_tokens == 0                    (waiting / fresh)

_advance_running() brings num_tokens up to total_len (by reserving a
block slot for the most recent output token). _admit_waiting() admits
fresh sequences by reserving and advancing num_tokens up to prompt_len
in one shot.

Preemption policy
-----------------
Recompute, not swap. When we evict a sequence, its blocks are returned
to the pool and it goes back to the FRONT of the waiting queue (so it
gets re-admitted as soon as there is room again). On re-admission we
prefill over (prompt + output-so-far); the generated tokens so far are
preserved on the Sequence object so the re-prefill sees the full history.

Why LIFO preemption (evict newest-admitted first)?
    Sequences admitted most recently have generated the fewest tokens,
    so recomputing their KV on re-admission is cheapest. Evicting a
    long-running sequence would waste more work.

Why appendleft on re-admission (preempted goes to front of waiting)?
    Otherwise the newly-admitted sequence could starve it indefinitely
    as new requests keep arriving at the back of the queue.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List

from src.kv_cache.allocator import BlockAllocator
from src.scheduler.sequence import Sequence, SequenceStatus


@dataclass
class SchedulerConfig:
    block_size: int = 16
    max_num_seqs: int = 32
    # Upper bound on (decoders + total prefill tokens) per step. Keeps
    # peak memory on the model side predictable once a real model is
    # wired in. For the dummy model it is effectively inert as long as
    # the cap is larger than the longest prompt.
    max_num_batched_tokens: int = 4096


@dataclass
class SchedulerOutput:
    prefill_seqs: List[Sequence] = field(default_factory=list)
    decode_seqs: List[Sequence] = field(default_factory=list)
    preempted_seqs: List[Sequence] = field(default_factory=list)

    @property
    def num_batched(self) -> int:
        return len(self.prefill_seqs) + len(self.decode_seqs)

    @property
    def is_empty(self) -> bool:
        return self.num_batched == 0

    def iter_seqs(self):
        yield from self.prefill_seqs
        yield from self.decode_seqs


class Scheduler:
    def __init__(self, allocator: BlockAllocator, config: SchedulerConfig) -> None:
        self._allocator = allocator
        self._config = config
        self._waiting: Deque[Sequence] = deque()
        self._running: List[Sequence] = []
        self._finished: List[Sequence] = []

    # ------------------------------------------------------------------
    # Queue introspection
    # ------------------------------------------------------------------

    @property
    def num_waiting(self) -> int:
        return len(self._waiting)

    @property
    def num_running(self) -> int:
        return len(self._running)

    @property
    def num_finished(self) -> int:
        return len(self._finished)

    @property
    def num_free_blocks(self) -> int:
        return self._allocator.num_free

    @property
    def has_work(self) -> bool:
        return bool(self._waiting) or bool(self._running)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add_sequence(self, seq: Sequence) -> None:
        seq.status = SequenceStatus.WAITING
        self._waiting.append(seq)

    def step(self) -> SchedulerOutput:
        output = SchedulerOutput()
        output.preempted_seqs = self._advance_running()
        admitted = self._admit_waiting()
        output.prefill_seqs = admitted
        admitted_ids = {id(s) for s in admitted}
        output.decode_seqs = [s for s in self._running if id(s) not in admitted_ids]
        return output

    def commit(
        self, output: SchedulerOutput, next_tokens: Dict[str, int]
    ) -> List[Sequence]:
        """
        Apply the model's output: append the new token to each running
        sequence, mark any that hit their stop condition as finished,
        and free their blocks. Returns the list of sequences that
        finished this step.
        """
        finished: List[Sequence] = []
        still_running: List[Sequence] = []
        for seq in self._running:
            if seq.request_id not in next_tokens:
                raise KeyError(
                    f"model did not produce a token for request {seq.request_id!r}"
                )
            seq.append_output(next_tokens[seq.request_id])
            if seq.output_len >= seq.max_new_tokens:
                seq.status = SequenceStatus.FINISHED_LENGTH
                seq.block_table.free()
                finished.append(seq)
                self._finished.append(seq)
            else:
                still_running.append(seq)
        self._running = still_running
        return finished

    # ------------------------------------------------------------------
    # Step pass 1: advance running sequences by one token of KV reservation
    # ------------------------------------------------------------------

    def _advance_running(self) -> List[Sequence]:
        """
        For each currently-running seq, bring its KV-cache pointer up to
        its logical length (num_tokens -> total_len). Preempt LIFO on OOM.
        """
        preempted: List[Sequence] = []
        survivors: List[Sequence] = []

        for seq in self._running:
            needed_tokens = seq.total_len - seq.block_table.num_tokens
            if needed_tokens < 0:
                raise RuntimeError(
                    f"seq {seq.request_id!r}: num_tokens "
                    f"({seq.block_table.num_tokens}) exceeds total_len "
                    f"({seq.total_len})"
                )
            if needed_tokens == 0:
                survivors.append(seq)
                continue

            needed_blocks = seq.block_table.blocks_needed_for(needed_tokens)
            evicted_self = False

            # Preempt LIFO from already-survived seqs until the reservation fits.
            while needed_blocks > 0 and not self._allocator.can_allocate(needed_blocks):
                if survivors:
                    victim = survivors.pop()
                else:
                    # Nothing else to give up; the sequence we were trying
                    # to grow has to evict itself.
                    self._preempt(seq)
                    preempted.append(seq)
                    evicted_self = True
                    break
                self._preempt(victim)
                preempted.append(victim)

            if evicted_self:
                continue

            seq.block_table.allocate_for(needed_tokens)
            seq.block_table.append_tokens(needed_tokens)
            survivors.append(seq)

        self._running = survivors
        return preempted

    # ------------------------------------------------------------------
    # Step pass 2: admit new sequences from the waiting queue
    # ------------------------------------------------------------------

    def _admit_waiting(self) -> List[Sequence]:
        admitted: List[Sequence] = []
        # Decoders already consume 1 slot each in the token budget; any
        # new prefill adds its full length.
        tokens_budget = self._config.max_num_batched_tokens - len(self._running)

        while self._waiting:
            if len(self._running) >= self._config.max_num_seqs:
                break
            seq = self._waiting[0]
            if seq.block_table.num_tokens != 0:
                # Defensive: sequences on the waiting queue should always
                # have an empty block table (fresh or just-preempted).
                break

            prompt_kv_len = seq.total_len  # prompt + any previously generated tokens
            if prompt_kv_len > tokens_budget:
                break

            needed_blocks = seq.block_table.blocks_needed_for(prompt_kv_len)
            if not self._allocator.can_allocate(needed_blocks):
                break

            self._waiting.popleft()
            seq.block_table.allocate_for(prompt_kv_len)
            seq.block_table.append_tokens(prompt_kv_len)
            seq.status = SequenceStatus.RUNNING
            self._running.append(seq)
            admitted.append(seq)
            tokens_budget -= prompt_kv_len

        return admitted

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _preempt(self, seq: Sequence) -> None:
        """
        Free a sequence's KV blocks and send it to the front of the
        waiting queue for re-prefill on a later step. Any tokens the
        sequence has already generated stay on the Sequence object so
        that re-admission can prefill over the full history.
        """
        seq.block_table.free()
        seq.status = SequenceStatus.WAITING
        self._waiting.appendleft(seq)
