"""
Per-sequence mapping from logical token positions to physical KV blocks.

For a sequence with logical tokens 0..N-1 and a block size of B, logical
position i lives in physical block `block_table[i // B]` at slot `i % B`.
This indirection is what makes "paging" work: each sequence is free to
grow to any length the block pool can accommodate without ever needing a
contiguous KV region.

Growth model:
    allocate_for(k)   reserves enough new physical blocks to hold k more
                      tokens on top of the current logical length
    append_tokens(k)  advances the logical write pointer by k tokens
                      (the caller must have reserved room first)
    free()            returns every physical block to the pool

Keeping allocation and pointer-advancement as separate operations lets
the scheduler reserve room eagerly (so it can fail early and preempt)
and have the write pointer advance only after the model has actually
produced the data.

        logical tokens: 0 1 2 3 | 4 5 6 7 | 8 9
                        [block A] [block B] [C]
        block_table   =  [A, B, C]
        num_tokens    =  10
        num_slots_free_in_tail = block_size - (10 - 2*B) = B - 2
"""

from __future__ import annotations

from typing import List

from .allocator import BlockAllocator


class BlockTable:
    def __init__(self, allocator: BlockAllocator, block_size: int) -> None:
        if block_size <= 0:
            raise ValueError(f"block_size must be positive, got {block_size}")
        self._allocator = allocator
        self._block_size = block_size
        self._blocks: List[int] = []
        self._num_tokens: int = 0

    @property
    def block_size(self) -> int:
        return self._block_size

    @property
    def num_tokens(self) -> int:
        return self._num_tokens

    @property
    def num_blocks(self) -> int:
        return len(self._blocks)

    @property
    def physical_blocks(self) -> List[int]:
        """Read-only copy of the current physical block ids."""
        return list(self._blocks)

    def num_slots_free_in_tail(self) -> int:
        """
        How many unused token slots live in the last allocated block.
        If there is no tail block at all, returns 0.
        """
        if not self._blocks:
            return 0
        used_in_tail = self._num_tokens - (len(self._blocks) - 1) * self._block_size
        return self._block_size - used_in_tail

    def blocks_needed_for(self, extra_tokens: int) -> int:
        """
        How many NEW physical blocks would be required to fit `extra_tokens`
        more tokens on top of the current occupancy.
        """
        if extra_tokens <= 0:
            return 0
        free_in_tail = self.num_slots_free_in_tail()
        remaining = max(0, extra_tokens - free_in_tail)
        return (remaining + self._block_size - 1) // self._block_size

    def allocate_for(self, extra_tokens: int) -> None:
        """
        Reserve physical blocks to cover `extra_tokens` more tokens. Does
        NOT advance the write pointer; call append_tokens() for that.
        """
        needed = self.blocks_needed_for(extra_tokens)
        if needed == 0:
            return
        new_ids = self._allocator.allocate(needed)
        self._blocks.extend(new_ids)

    def append_tokens(self, n: int) -> None:
        """
        Advance the logical write pointer by n. The caller must have
        previously reserved space via allocate_for().
        """
        if n < 0:
            raise ValueError(f"n must be non-negative, got {n}")
        if n == 0:
            return
        new_total = self._num_tokens + n
        blocks_required = (new_total + self._block_size - 1) // self._block_size
        if blocks_required > len(self._blocks):
            raise RuntimeError(
                f"append_tokens({n}) would need {blocks_required} blocks "
                f"but only {len(self._blocks)} are reserved; "
                f"call allocate_for() first"
            )
        self._num_tokens = new_total

    def free(self) -> None:
        """Return all physical blocks to the pool. Idempotent."""
        if not self._blocks:
            self._num_tokens = 0
            return
        self._allocator.free(self._blocks)
        self._blocks = []
        self._num_tokens = 0

    def __repr__(self) -> str:
        return (
            f"BlockTable(num_tokens={self._num_tokens}, "
            f"num_blocks={len(self._blocks)}, block_size={self._block_size})"
        )
