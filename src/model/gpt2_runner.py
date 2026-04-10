"""
Bridge between the Scheduler and the GPT2PagedModel.

The scheduler speaks in Sequences (with block tables, output buffers,
and request ids); the model speaks in tensors. The runner converts.

Per scheduler step the runner:
  1. Walks every prefill sequence one at a time, running a forward over
     its full (prompt + any recompute output) token stream.
  2. Batches all decode sequences into a single model call:
     embeddings, linear projections, LayerNorms, and MLP are computed
     in one fused [B, D] matmul; only the per-sequence paged attention
     loops per-sequence inside the model.
  3. Argmax-samples each sequence's logits and returns the
     {request_id -> next_token_id} map for the scheduler's commit().

Prefill stays one-at-a-time because variable-length prefills would need
packed varlen (cu_seqlens), which gains nothing with eager / per-
sequence attention and is deferred to a future batched-prefill kernel.
"""

from __future__ import annotations

from typing import Dict, List

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

        # -- Prefill: one at a time (variable-length; no GPU batching) --
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

        # -- Decode: batched across all decode sequences --
        if output.decode_seqs:
            next_tokens.update(self._batched_decode(output.decode_seqs))

        return next_tokens

    def _batched_decode(self, seqs) -> Dict[str, int]:
        """
        Stack all decode sequences into one model call. Block tables are
        padded into a single [B, max_blocks] tensor built once and shared
        across all 12 layers. The batched write and batched paged decode
        kernel each launch once per layer instead of B times.
        """
        B = len(seqs)
        token_ids: List[int] = []
        pos_list: List[int] = []
        ctx_lens: List[int] = []
        block_id_lists: List[List[int]] = []

        for seq in seqs:
            token_ids.append(seq.output_token_ids[-1])
            pos = seq.total_len - 1
            pos_list.append(pos)
            ctx_lens.append(pos + 1)
            block_id_lists.append(seq.block_table.physical_blocks)

        input_ids = torch.tensor(token_ids, dtype=torch.long, device=self._device)
        positions = torch.tensor(pos_list, dtype=torch.long, device=self._device)
        context_lens = torch.tensor(ctx_lens, dtype=torch.int32, device=self._device)

        # Pad block tables to the same length and pack into [B, max_blocks].
        max_blocks = max(len(bl) for bl in block_id_lists)
        padded: List[List[int]] = []
        for bl in block_id_lists:
            padded.append(bl + [0] * (max_blocks - len(bl)))
        block_tables_2d = torch.tensor(
            padded, dtype=torch.int32, device=self._device
        )

        logits = self._model.decode_batch(
            input_ids, positions, block_tables_2d, context_lens, self._kv_pool
        )  # [B, V]

        tokens: Dict[str, int] = {}
        for i, seq in enumerate(seqs):
            tokens[seq.request_id] = int(logits[i].argmax().item())
        return tokens
