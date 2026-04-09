import pytest

from src.kv_cache.allocator import BlockAllocator
from src.kv_cache.block_table import BlockTable


def test_initial_state() -> None:
    a = BlockAllocator(4)
    t = BlockTable(a, block_size=4)
    assert t.num_tokens == 0
    assert t.num_blocks == 0
    assert t.physical_blocks == []
    assert t.num_slots_free_in_tail() == 0
    assert t.block_size == 4


def test_block_size_must_be_positive() -> None:
    a = BlockAllocator(4)
    with pytest.raises(ValueError):
        BlockTable(a, block_size=0)
    with pytest.raises(ValueError):
        BlockTable(a, block_size=-2)


def test_blocks_needed_empty_table() -> None:
    a = BlockAllocator(10)
    t = BlockTable(a, block_size=4)
    assert t.blocks_needed_for(0) == 0
    assert t.blocks_needed_for(1) == 1
    assert t.blocks_needed_for(4) == 1
    assert t.blocks_needed_for(5) == 2
    assert t.blocks_needed_for(9) == 3
    assert t.blocks_needed_for(-3) == 0


def test_allocate_and_append_basic() -> None:
    a = BlockAllocator(10)
    t = BlockTable(a, block_size=4)
    t.allocate_for(10)
    assert t.num_blocks == 3
    assert a.num_free == 7
    t.append_tokens(10)
    assert t.num_tokens == 10
    # 10 tokens across 3 blocks of size 4: last block holds 2, so 2 slots free.
    assert t.num_slots_free_in_tail() == 2


def test_append_without_allocate_raises() -> None:
    a = BlockAllocator(10)
    t = BlockTable(a, block_size=4)
    with pytest.raises(RuntimeError):
        t.append_tokens(1)


def test_append_zero_is_noop() -> None:
    a = BlockAllocator(10)
    t = BlockTable(a, block_size=4)
    t.append_tokens(0)
    assert t.num_tokens == 0


def test_append_negative_raises() -> None:
    a = BlockAllocator(10)
    t = BlockTable(a, block_size=4)
    with pytest.raises(ValueError):
        t.append_tokens(-1)


def test_growing_across_block_boundary() -> None:
    a = BlockAllocator(10)
    t = BlockTable(a, block_size=4)

    # Fill first block exactly.
    t.allocate_for(4)
    t.append_tokens(4)
    assert t.num_blocks == 1
    assert t.num_slots_free_in_tail() == 0

    # The very next token requires a new block.
    assert t.blocks_needed_for(1) == 1
    t.allocate_for(1)
    t.append_tokens(1)
    assert t.num_blocks == 2
    assert t.num_tokens == 5
    assert t.num_slots_free_in_tail() == 3


def test_free_returns_all_blocks() -> None:
    a = BlockAllocator(10)
    t = BlockTable(a, block_size=4)
    t.allocate_for(7)
    t.append_tokens(7)
    assert a.num_free == 8  # 10 - ceil(7/4) = 10 - 2 = 8
    t.free()
    assert a.num_free == 10
    assert t.num_tokens == 0
    assert t.num_blocks == 0
    assert t.physical_blocks == []


def test_free_is_idempotent() -> None:
    a = BlockAllocator(10)
    t = BlockTable(a, block_size=4)
    t.allocate_for(3)
    t.append_tokens(3)
    t.free()
    t.free()  # should not raise or affect state
    assert a.num_free == 10


def test_multiple_tables_share_one_pool() -> None:
    """Two sequences on one allocator do not collide and can each be freed."""
    a = BlockAllocator(8)
    t1 = BlockTable(a, block_size=2)
    t2 = BlockTable(a, block_size=2)

    t1.allocate_for(6)  # 3 blocks
    t1.append_tokens(6)
    t2.allocate_for(4)  # 2 blocks
    t2.append_tokens(4)
    assert a.num_free == 3
    # Physical ids must be disjoint.
    assert set(t1.physical_blocks).isdisjoint(t2.physical_blocks)

    t1.free()
    assert a.num_free == 6
    t2.free()
    assert a.num_free == 8


def test_allocate_for_propagates_oom() -> None:
    from src.kv_cache.allocator import OutOfBlocksError

    a = BlockAllocator(2)
    t = BlockTable(a, block_size=4)
    with pytest.raises(OutOfBlocksError):
        t.allocate_for(100)  # would need 25 blocks
    # Pool must be untouched.
    assert a.num_free == 2
