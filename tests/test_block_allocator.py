import pytest

from src.kv_cache.allocator import BlockAllocator, OutOfBlocksError


def test_construct_requires_positive_blocks() -> None:
    with pytest.raises(ValueError):
        BlockAllocator(0)
    with pytest.raises(ValueError):
        BlockAllocator(-1)


def test_initial_state() -> None:
    a = BlockAllocator(8)
    assert a.num_total == 8
    assert a.num_free == 8
    assert a.num_allocated == 0
    assert a.can_allocate(8)
    assert not a.can_allocate(9)


def test_allocate_and_free_roundtrip() -> None:
    a = BlockAllocator(8)
    ids = a.allocate(3)
    assert len(ids) == 3
    assert len(set(ids)) == 3  # no duplicate ids handed out
    assert a.num_free == 5
    assert a.num_allocated == 3
    a.free(ids)
    assert a.num_free == 8
    assert a.num_allocated == 0


def test_allocate_zero_is_noop() -> None:
    a = BlockAllocator(4)
    assert a.allocate(0) == []
    assert a.num_free == 4


def test_allocate_negative_raises() -> None:
    a = BlockAllocator(4)
    with pytest.raises(ValueError):
        a.allocate(-1)


def test_out_of_blocks_raises() -> None:
    a = BlockAllocator(4)
    a.allocate(4)
    assert a.num_free == 0
    with pytest.raises(OutOfBlocksError):
        a.allocate(1)


def test_can_allocate() -> None:
    a = BlockAllocator(4)
    assert a.can_allocate(0)
    assert a.can_allocate(4)
    assert not a.can_allocate(5)
    a.allocate(3)
    assert a.can_allocate(1)
    assert not a.can_allocate(2)


def test_free_list_recycles() -> None:
    a = BlockAllocator(4)
    first = a.allocate(4)
    a.free(first)
    second = a.allocate(4)
    # Every id we freed must be handed back out on the next full alloc.
    assert sorted(first) == sorted(second)


def test_double_free_raises_and_state_unchanged() -> None:
    a = BlockAllocator(4)
    ids = a.allocate(2)
    a.free(ids)
    before_free = a.num_free
    with pytest.raises(ValueError):
        a.free(ids)
    assert a.num_free == before_free


def test_free_unallocated_id_raises() -> None:
    a = BlockAllocator(4)
    a.allocate(2)
    with pytest.raises(ValueError):
        a.free([99])


def test_free_rejects_partial_bad_batch_before_mutating() -> None:
    """
    If a batch free contains one bad id, the allocator must reject the
    whole batch without freeing any of the good ids. Otherwise a caller
    retrying the same batch would double-free the good ids.
    """
    a = BlockAllocator(4)
    good = a.allocate(2)
    before = a.num_free
    with pytest.raises(ValueError):
        a.free([good[0], 999])
    assert a.num_free == before
    assert a.num_allocated == 2


def test_fragmentation_pattern_alternating_free() -> None:
    a = BlockAllocator(10)
    ids = a.allocate(10)
    even_ids = [b for i, b in enumerate(ids) if i % 2 == 0]
    a.free(even_ids)
    assert a.num_free == 5
    assert a.num_allocated == 5
    reclaimed = a.allocate(5)
    assert a.num_free == 0
    # Re-allocated ids must be a subset of what we freed (the allocator
    # never invents new ids).
    assert set(reclaimed) == set(even_ids)


def test_stress_alloc_free_preserves_invariants() -> None:
    """Hammer alloc/free in a mixed sequence; invariants must hold at every step."""
    a = BlockAllocator(64)
    held: list[int] = []
    for i in range(500):
        if i % 3 == 0 and len(held) >= 4:
            victims = held[:4]
            held = held[4:]
            a.free(victims)
        else:
            try:
                held.extend(a.allocate(5))
            except OutOfBlocksError:
                # expected when nearly full; fall through
                pass
        assert a.num_free + a.num_allocated == a.num_total
        assert a.num_allocated == len(held)
    a.free(held)
    assert a.num_free == a.num_total
