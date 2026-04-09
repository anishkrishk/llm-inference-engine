"""
Bridge between the Scheduler and the GPT2PagedModel.

The scheduler speaks in Sequences (with block tables, output buffers,
and request ids); the model speaks in tensors. The runner converts.

Per scheduler step the runner:
  1. Walks every prefill sequence, runs a forward over its full
     (prompt + any previously generated output) token stream, and
     argmax-samples the logits at the final position for the next token.
  2. Walks every decode sequence, runs a forward over just its single
     most-recent generated token, and argmax-samples the next token.
  3. Returns a {request_id -> next_token_id} map for the scheduler's
     commit() call.

Sequences are processed one at a time (no GPU batching) in this phase.
The eager attention path doesn't benefit from packing, and correctness
is easier to verify one sequence at a time. A batched attention kernel
is the thing that would actually exploit a packed varlen layout; that
lands in the Triton-kernel phase.
"""

from __future__ import annotations

from typing import Dict

import torch

from src.kv_cache.kv_pool import KVPool
from src.model.gpt2 import GPT2PagedModel
from src.scheduler.scheduler import SchedulerOutput


class GPT2Runner:
    def __init__(self, model: GPT2PagedModel, kv_pool: KVPool) -> None:
        self._model = model
        self._kv_pool = kv_pool
        self._device = next(model.parameters()).device

    @torch.inference_mode()
    def step(self, output: SchedulerOutput) -> Dict[str, int]:
        next_tokens: Dict[str, int] = {}

        # Prefill: process the full (prompt + any recompute output) at
        # start_pos = 0.
        for seq in output.prefill_seqs:
            ids = torch.as_tensor(
                seq.all_token_ids(), dtype=torch.long, device=self._device
            )
            logits = self._model(
                input_ids=ids,
                start_pos=0,
                block_ids=seq.block_table.physical_blocks,
                kv_pool=self._kv_pool,
            )
            next_tokens[seq.request_id] = int(logits[-1].argmax().item())

        # Decode: one token, one position, attending over the cache.
        for seq in output.decode_seqs:
            last_token = seq.output_token_ids[-1]
            # The scheduler has already reserved a slot for this new
            # token in its block table via Phase A's append_tokens().
            # seq.total_len == num_tokens in the block table, and the
            # new token lives at position (total_len - 1).
            start_pos = seq.total_len - 1
            ids = torch.as_tensor(
                [last_token], dtype=torch.long, device=self._device
            )
            logits = self._model(
                input_ids=ids,
                start_pos=start_pos,
                block_ids=seq.block_table.physical_blocks,
                kv_pool=self._kv_pool,
            )
            next_tokens[seq.request_id] = int(logits[-1].argmax().item())

        return next_tokens
