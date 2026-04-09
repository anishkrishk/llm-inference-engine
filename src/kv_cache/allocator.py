"""
Fixed-pool block allocator with a free list.

The allocator owns integer block ids and nothing else. The scheduler,
sequences, and block tables speak in ids; the ids will eventually index
into a KV tensor living in some other module. Decoupling ownership this
way keeps the allocator testable without CUDA and makes the bookkeeping
easy to reason about.

Semantics:
    allocate(n)  -> List[int]   # raises OutOfBlocksError on failure
    free(ids)    -> None        # raises ValueError on double/invalid free

Policy: LIFO free list. Freshly freed blocks are reused first. Two
reasons we pick LIFO over FIFO:
  1. Under preemption cycles the "hot" working set of blocks stays small,
     which gives better cache behavior once a real kernel touches the
     backing tensor.
  2. It makes test assertions about which ids come back from allocate()
     deterministic given a known history of free() calls.

This module has no torch dependency by design.
"""

from __future__ import annotations

from collections import deque
from typing import Deque, Iterable, List, Set


class OutOfBlocksError(RuntimeError):
    """Raised when the block pool cannot satisfy an allocation request."""


class BlockAllocator:
    def __init__(self, num_blocks: int) -> None:
        if num_blocks <= 0:
            raise ValueError(f"num_blocks must be positive, got {num_blocks}")
        self._num_blocks = num_blocks
        # deque used as a LIFO stack: pop() from the right end.
        self._free: Deque[int] = deque(range(num_blocks))
        self._allocated: Set[int] = set()

    @property
    def num_total(self) -> int:
        return self._num_blocks

    @property
    def num_free(self) -> int:
        return len(self._free)

    @property
    def num_allocated(self) -> int:
        return len(self._allocated)

    def can_allocate(self, n: int) -> bool:
        return 0 <= n <= len(self._free)

    def allocate(self, n: int = 1) -> List[int]:
        if n < 0:
            raise ValueError(f"cannot allocate {n} blocks (negative)")
        if n == 0:
            return []
        if n > len(self._free):
            raise OutOfBlocksError(
                f"requested {n} blocks, only {len(self._free)} free "
                f"(out of {self._num_blocks} total)"
            )
        out: List[int] = []
        for _ in range(n):
            bid = self._free.pop()  # LIFO
            self._allocated.add(bid)
            out.append(bid)
        return out

    def free(self, ids: Iterable[int]) -> None:
        # Materialize up front so a bad id aborts before we mutate state.
        id_list = list(ids)
        for bid in id_list:
            if bid not in self._allocated:
                raise ValueError(
                    f"cannot free block {bid}: not currently allocated "
                    f"(double free or invalid id)"
                )
        for bid in id_list:
            self._allocated.remove(bid)
            self._free.append(bid)

    def __repr__(self) -> str:
        return (
            f"BlockAllocator(total={self._num_blocks}, "
            f"free={len(self._free)}, allocated={len(self._allocated)})"
        )
