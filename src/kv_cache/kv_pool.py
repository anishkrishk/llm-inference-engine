"""
Tensor-backed KV cache that lives alongside the integer-only BlockAllocator.

The BlockAllocator in `src.kv_cache.allocator` deals only in block ids; it
has no idea those ids point to memory. The KVPool is the other half of
that split: it owns the actual K and V tensors and knows how to write
into and read out of them using the same ids the allocator hands out.

Layout
------
Per layer we keep two tensors:

    k_cache: [num_layers, num_blocks, block_size, num_heads, head_dim]
    v_cache: [num_layers, num_blocks, block_size, num_heads, head_dim]

Why this shape:

  * `num_layers` first lets the model grab a single layer's slab with a
    cheap slice (`k_cache[layer_idx]`). Attention for layer L only ever
    touches layer L's slab.
  * `num_blocks` next means gathering a sequence's blocks is one line of
    advanced indexing: `layer_slab[torch.tensor(block_ids)]`.
  * `block_size` inside `num_blocks` keeps one block contiguous in
    memory. When the eventual decode kernel loads a block it will get
    coalesced memory reads.
  * `num_heads` then `head_dim` matches the attention code's expected
    shape so we can reshape without permute().

Keeping K and V as two separate tensors (rather than stacking them on a
leading `2` axis) makes the indexing calls a little less fiddly and lets
us pass them around independently.

The pool is fixed-size: allocation happens once at startup, and from
then on writes and reads just index into pre-allocated memory. There is
no dynamic resize path. If a sequence wants more blocks, the
BlockAllocator hands out ids from within this fixed pool, and if the
pool is full, the scheduler preempts.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence as SequenceT
from typing import Tuple

import torch


@dataclass(frozen=True)
class KVPoolConfig:
    num_layers: int
    num_blocks: int
    block_size: int
    num_heads: int
    head_dim: int
    dtype: torch.dtype
    device: torch.device

    def bytes_per_block(self) -> int:
        elt = torch.tensor([], dtype=self.dtype).element_size()
        # per block, per layer, 2 (K+V), block_size * heads * head_dim elements
        return 2 * self.block_size * self.num_heads * self.head_dim * elt

    def total_bytes(self) -> int:
        return self.num_layers * self.num_blocks * self.bytes_per_block()


class KVPool:
    """
    Owns the backing K / V tensors for a paged KV cache.

    All positions below use the sequence's *logical* position — that is,
    the token's index within its sequence starting from 0. The block
    table maps logical position -> physical block; the pool only sees
    physical block ids and resolves them into tensor indices.
    """

    def __init__(self, config: KVPoolConfig, memory_guard_fraction: float = 0.5) -> None:
        self._cfg = config

        # Simple sanity guard: refuse to allocate a pool that would eat
        # more than `memory_guard_fraction` of the currently-free VRAM.
        # Prevents accidental OOM during engine construction when someone
        # asks for a pool far bigger than the card can hold.
        if config.device.type == "cuda":
            free_bytes, _total_bytes = torch.cuda.mem_get_info(config.device)
            budget = int(free_bytes * memory_guard_fraction)
            needed = config.total_bytes()
            if needed > budget:
                raise RuntimeError(
                    f"KVPool would require {needed / 1e6:.1f} MB but only "
                    f"{budget / 1e6:.1f} MB is within the {memory_guard_fraction:.0%} "
                    f"memory guard ({free_bytes / 1e6:.1f} MB free). "
                    f"Reduce num_blocks or num_layers."
                )

        shape = (
            config.num_layers,
            config.num_blocks,
            config.block_size,
            config.num_heads,
            config.head_dim,
        )
        self._k = torch.empty(shape, dtype=config.dtype, device=config.device)
        self._v = torch.empty(shape, dtype=config.dtype, device=config.device)

    @property
    def config(self) -> KVPoolConfig:
        return self._cfg

    @property
    def k_cache(self) -> torch.Tensor:
        return self._k

    @property
    def v_cache(self) -> torch.Tensor:
        return self._v

    # ------------------------------------------------------------------
    # Write path
    # ------------------------------------------------------------------

    def write(
        self,
        layer_idx: int,
        block_ids,
        start_pos: int,
        k_new: torch.Tensor,
        v_new: torch.Tensor,
    ) -> None:
        """
        Write K/V for positions [start_pos, start_pos + L) of a sequence.

        Arguments
        ---------
        layer_idx:  which transformer layer's slab to write into.
        block_ids:  the block table — a Python list/sequence of ints OR
                    a 1-D torch.Tensor already on the same device.
        start_pos:  logical position of the first new token.
        k_new:      tensor of shape [L, num_heads, head_dim]
        v_new:      tensor of shape [L, num_heads, head_dim]

        The write is vectorized: we compute the (physical_block, slot)
        pair for each of the L positions and scatter-assign in one op.
        """
        L = k_new.shape[0]
        if L == 0:
            return
        if v_new.shape[0] != L:
            raise ValueError(
                f"k_new and v_new must have same length, got {L} vs {v_new.shape[0]}"
            )

        block_size = self._cfg.block_size
        device = self._k.device

        # positions[i] = logical position of the i-th new token
        positions = torch.arange(start_pos, start_pos + L, device=device)
        logical_block = positions // block_size    # [L]
        slot = positions % block_size              # [L]

        max_logical = int(logical_block.max().item()) + 1

        # Accept either a Python list or a CUDA tensor for block_ids.
        # The batched runner passes pre-built tensors to avoid repeated
        # CPU->GPU transfers across layers.
        if isinstance(block_ids, torch.Tensor):
            if block_ids.shape[0] < max_logical:
                raise IndexError(
                    f"write would touch logical block {max_logical - 1} but the "
                    f"block table only has {block_ids.shape[0]} entries"
                )
            block_table_tensor = block_ids[:max_logical].to(
                device=device, dtype=torch.long
            )
        else:
            if max_logical > len(block_ids):
                raise IndexError(
                    f"write would touch logical block {max_logical - 1} but the "
                    f"block table only has {len(block_ids)} entries"
                )
            block_table_tensor = torch.as_tensor(
                list(block_ids[:max_logical]), device=device, dtype=torch.long
            )
        physical = block_table_tensor[logical_block]   # [L]

        # Advanced indexing scatter write. The three index tensors are
        # broadcast together; assignment fills the matching slots.
        self._k[layer_idx, physical, slot] = k_new
        self._v[layer_idx, physical, slot] = v_new

    def write_decode_batch(
        self,
        layer_idx: int,
        block_tables: torch.Tensor,
        positions: torch.Tensor,
        k_batch: torch.Tensor,
        v_batch: torch.Tensor,
    ) -> None:
        """
        Scatter-write one KV pair per sequence for a batch of B sequences.

        This replaces B individual write() calls with a single vectorized
        scatter, eliminating B-1 Python call frames per layer per step.

        Args
        ----
        layer_idx    : transformer layer index
        block_tables : [B, max_blocks] int32 CUDA tensor (physical block ids)
        positions    : [B] int64 CUDA tensor (the position to write each seq)
        k_batch      : [B, H, D] fp16 CUDA tensor (one token per seq)
        v_batch      : [B, H, D] fp16 CUDA tensor
        """
        B = k_batch.shape[0]
        block_size = self._cfg.block_size
        logical_block = positions // block_size  # [B]
        slot = positions % block_size            # [B]

        # Gather each sequence's physical block id from its row of block_tables.
        batch_idx = torch.arange(B, device=positions.device)
        physical = block_tables[batch_idx, logical_block.long()].long()  # [B]

        # Scatter write: one slot per sequence, all in one advanced-indexing op.
        self._k[layer_idx, physical, slot] = k_batch
        self._v[layer_idx, physical, slot] = v_batch

    # ------------------------------------------------------------------
    # Read path
    # ------------------------------------------------------------------

    def read(
        self,
        layer_idx: int,
        block_ids: SequenceT[int],
        num_tokens: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Read the first `num_tokens` cached K/V for a sequence.

        Returns
        -------
        (K, V), each of shape [num_tokens, num_heads, head_dim]
        """
        if num_tokens == 0:
            empty = self._k.new_empty((0, self._cfg.num_heads, self._cfg.head_dim))
            return empty, empty

        block_size = self._cfg.block_size
        num_logical = (num_tokens + block_size - 1) // block_size
        if num_logical > len(block_ids):
            raise IndexError(
                f"read of {num_tokens} tokens needs {num_logical} logical "
                f"blocks but block_table only has {len(block_ids)}"
            )
        relevant = list(block_ids[:num_logical])
        idx = torch.as_tensor(relevant, device=self._k.device, dtype=torch.long)

        # [num_logical, block_size, num_heads, head_dim]
        k_blocks = self._k[layer_idx].index_select(0, idx)
        v_blocks = self._v[layer_idx].index_select(0, idx)

        # Flatten block + slot axes to a single "position" axis.
        k_flat = k_blocks.reshape(-1, self._cfg.num_heads, self._cfg.head_dim)
        v_flat = v_blocks.reshape(-1, self._cfg.num_heads, self._cfg.head_dim)

        # Trim away any padding beyond num_tokens (the final block may be
        # only partially filled).
        return k_flat[:num_tokens], v_flat[:num_tokens]
