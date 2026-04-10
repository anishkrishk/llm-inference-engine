"""
Correctness tests for the paged decode-attention kernel.

The reference implementation gathers K/V from the pool via
``KVPool.read()`` and runs eager attention; the kernel is expected
to match it within fp16 tolerance.

Critical case: ``test_non_contiguous_block_table``. The whole point
of paged attention is that physical block ids may be in any order;
if the kernel mishandles the indirection, this test catches it.
"""

from __future__ import annotations

import math

import pytest
import torch

cuda_available = torch.cuda.is_available()
pytestmark = pytest.mark.skipif(
    not cuda_available, reason="paged decode tests require CUDA"
)


def _reference_decode(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    block_table_list: list[int],
    context_len: int,
    layer_idx: int,
) -> torch.Tensor:
    """
    Eager reference: gather K, V from the paged cache for the given
    sequence and compute single-query attention.
    """
    from src.kv_cache.kv_pool import KVPool

    # Build a tiny KVPool view to reuse its read() helper. Avoiding the
    # construction by inlining the gather here would duplicate the
    # logic we want to test against.
    H, D = q.shape[1], q.shape[2]

    block_size = k_cache.shape[2]
    num_logical = (context_len + block_size - 1) // block_size
    relevant = block_table_list[:num_logical]
    idx = torch.tensor(relevant, dtype=torch.long, device=q.device)
    k_blocks = k_cache[layer_idx].index_select(0, idx)  # [n_logical, B, H, D]
    v_blocks = v_cache[layer_idx].index_select(0, idx)
    k_flat = k_blocks.reshape(-1, H, D)[:context_len]  # [ctx, H, D]
    v_flat = v_blocks.reshape(-1, H, D)[:context_len]

    scale = 1.0 / math.sqrt(D)
    q_h = q.transpose(0, 1)              # [H, 1, D]
    k_h = k_flat.transpose(0, 1)         # [H, ctx, D]
    v_h = v_flat.transpose(0, 1)         # [H, ctx, D]
    scores = torch.matmul(q_h, k_h.transpose(-1, -2)) * scale  # [H, 1, ctx]
    attn = torch.softmax(scores.float(), dim=-1).to(scores.dtype)
    out = torch.matmul(attn, v_h)        # [H, 1, D]
    return out.transpose(0, 1)            # [1, H, D]


def _setup_pool_with_random_kv(
    num_layers: int,
    num_blocks: int,
    block_size: int,
    num_heads: int,
    head_dim: int,
    block_table: list[int],
    context_len: int,
    seed: int = 0,
):
    """Build a KVPool and fill the requested blocks with random fp16 data."""
    from src.kv_cache.kv_pool import KVPool, KVPoolConfig

    device = torch.device("cuda")
    dtype = torch.float16
    pool = KVPool(
        KVPoolConfig(
            num_layers=num_layers,
            num_blocks=num_blocks,
            block_size=block_size,
            num_heads=num_heads,
            head_dim=head_dim,
            dtype=dtype,
            device=device,
        )
    )
    g = torch.Generator(device=device).manual_seed(seed)
    for layer in range(num_layers):
        k_data = torch.randn(
            context_len, num_heads, head_dim, dtype=dtype, device=device, generator=g
        )
        v_data = torch.randn(
            context_len, num_heads, head_dim, dtype=dtype, device=device, generator=g
        )
        pool.write(layer, block_table, start_pos=0, k_new=k_data, v_new=v_data)
    return pool


# ----------------------------------------------------------------------
# Correctness sweeps
# ----------------------------------------------------------------------


@pytest.mark.parametrize("context_len", [1, 8, 16, 17, 32, 100, 256, 512])
def test_paged_decode_matches_reference_contiguous(context_len: int) -> None:
    from src.kernels.triton.paged_decode import paged_decode_attention

    block_size = 16
    num_blocks_needed = (context_len + block_size - 1) // block_size
    block_table_list = list(range(num_blocks_needed))  # [0, 1, 2, ...]

    pool = _setup_pool_with_random_kv(
        num_layers=4,
        num_blocks=max(8, num_blocks_needed * 2),
        block_size=block_size,
        num_heads=12,
        head_dim=64,
        block_table=block_table_list,
        context_len=context_len,
        seed=context_len,
    )
    q = torch.randn(1, 12, 64, dtype=torch.float16, device="cuda")
    layer_idx = 2

    expected = _reference_decode(
        q, pool.k_cache, pool.v_cache, block_table_list, context_len, layer_idx
    )

    bt = torch.tensor(block_table_list, dtype=torch.int32, device="cuda")
    actual = paged_decode_attention(
        q, pool.k_cache, pool.v_cache, bt, context_len, layer_idx
    )

    torch.testing.assert_close(actual, expected, atol=5e-3, rtol=5e-2)


def test_non_contiguous_block_table() -> None:
    """The headline test: physical block ids in arbitrary order."""
    from src.kernels.triton.paged_decode import paged_decode_attention

    block_table_list = [7, 2, 11, 0, 5]  # deliberately scrambled
    block_size = 16
    context_len = 5 * block_size - 4  # 76 — last block partial
    pool = _setup_pool_with_random_kv(
        num_layers=4,
        num_blocks=16,
        block_size=block_size,
        num_heads=8,
        head_dim=64,
        block_table=block_table_list,
        context_len=context_len,
        seed=99,
    )
    q = torch.randn(1, 8, 64, dtype=torch.float16, device="cuda")

    expected = _reference_decode(
        q, pool.k_cache, pool.v_cache, block_table_list, context_len, layer_idx=1
    )
    bt = torch.tensor(block_table_list, dtype=torch.int32, device="cuda")
    actual = paged_decode_attention(
        q, pool.k_cache, pool.v_cache, bt, context_len, layer_idx=1
    )
    torch.testing.assert_close(actual, expected, atol=5e-3, rtol=5e-2)


@pytest.mark.parametrize("head_dim", [32, 64, 128])
def test_paged_decode_head_dim_variants(head_dim: int) -> None:
    from src.kernels.triton.paged_decode import paged_decode_attention

    block_table_list = [4, 1, 6]
    context_len = 40
    pool = _setup_pool_with_random_kv(
        num_layers=2,
        num_blocks=8,
        block_size=16,
        num_heads=8,
        head_dim=head_dim,
        block_table=block_table_list,
        context_len=context_len,
        seed=head_dim,
    )
    q = torch.randn(1, 8, head_dim, dtype=torch.float16, device="cuda")
    expected = _reference_decode(
        q, pool.k_cache, pool.v_cache, block_table_list, context_len, layer_idx=0
    )
    bt = torch.tensor(block_table_list, dtype=torch.int32, device="cuda")
    actual = paged_decode_attention(
        q, pool.k_cache, pool.v_cache, bt, context_len, layer_idx=0
    )
    torch.testing.assert_close(actual, expected, atol=5e-3, rtol=5e-2)


def test_two_dim_q_input_shape() -> None:
    """The wrapper accepts q as either [1, H, D] or [H, D]."""
    from src.kernels.triton.paged_decode import paged_decode_attention

    block_table_list = [0, 1]
    context_len = 20
    pool = _setup_pool_with_random_kv(
        num_layers=2, num_blocks=4, block_size=16,
        num_heads=4, head_dim=64,
        block_table=block_table_list, context_len=context_len, seed=0,
    )
    q3 = torch.randn(1, 4, 64, dtype=torch.float16, device="cuda")
    q2 = q3.squeeze(0)
    bt = torch.tensor(block_table_list, dtype=torch.int32, device="cuda")
    out3 = paged_decode_attention(q3, pool.k_cache, pool.v_cache, bt, context_len, layer_idx=0)
    out2 = paged_decode_attention(q2, pool.k_cache, pool.v_cache, bt, context_len, layer_idx=0)
    assert out3.shape == (1, 4, 64)
    assert out2.shape == (4, 64)
    torch.testing.assert_close(out3.squeeze(0), out2)


def test_int64_block_table_accepted() -> None:
    from src.kernels.triton.paged_decode import paged_decode_attention

    block_table_list = [3, 1]
    context_len = 25
    pool = _setup_pool_with_random_kv(
        num_layers=2, num_blocks=8, block_size=16,
        num_heads=4, head_dim=64,
        block_table=block_table_list, context_len=context_len, seed=0,
    )
    q = torch.randn(1, 4, 64, dtype=torch.float16, device="cuda")
    bt = torch.tensor(block_table_list, dtype=torch.int64, device="cuda")
    paged_decode_attention(q, pool.k_cache, pool.v_cache, bt, context_len, layer_idx=0)


# ----------------------------------------------------------------------
# Validation rejections
# ----------------------------------------------------------------------


def test_rejects_fp32() -> None:
    from src.kernels.triton.paged_decode import paged_decode_attention
    from src.kv_cache.kv_pool import KVPool, KVPoolConfig

    pool = KVPool(KVPoolConfig(
        num_layers=2, num_blocks=4, block_size=16, num_heads=4, head_dim=64,
        dtype=torch.float16, device=torch.device("cuda"),
    ))
    q = torch.randn(1, 4, 64, dtype=torch.float32, device="cuda")
    bt = torch.tensor([0], dtype=torch.int32, device="cuda")
    with pytest.raises(TypeError):
        paged_decode_attention(q, pool.k_cache, pool.v_cache, bt, 8, 0)


def test_rejects_zero_context_len() -> None:
    from src.kernels.triton.paged_decode import paged_decode_attention
    from src.kv_cache.kv_pool import KVPool, KVPoolConfig

    pool = KVPool(KVPoolConfig(
        num_layers=2, num_blocks=4, block_size=16, num_heads=4, head_dim=64,
        dtype=torch.float16, device=torch.device("cuda"),
    ))
    q = torch.randn(1, 4, 64, dtype=torch.float16, device="cuda")
    bt = torch.tensor([0], dtype=torch.int32, device="cuda")
    with pytest.raises(ValueError):
        paged_decode_attention(q, pool.k_cache, pool.v_cache, bt, 0, 0)


def test_rejects_block_table_too_short() -> None:
    from src.kernels.triton.paged_decode import paged_decode_attention
    from src.kv_cache.kv_pool import KVPool, KVPoolConfig

    pool = KVPool(KVPoolConfig(
        num_layers=2, num_blocks=8, block_size=16, num_heads=4, head_dim=64,
        dtype=torch.float16, device=torch.device("cuda"),
    ))
    q = torch.randn(1, 4, 64, dtype=torch.float16, device="cuda")
    bt = torch.tensor([0], dtype=torch.int32, device="cuda")
    # Only 1 block in table but context_len=20 needs 2 blocks at block_size=16
    with pytest.raises(ValueError):
        paged_decode_attention(q, pool.k_cache, pool.v_cache, bt, 20, 0)


def test_rejects_layer_out_of_range() -> None:
    from src.kernels.triton.paged_decode import paged_decode_attention
    from src.kv_cache.kv_pool import KVPool, KVPoolConfig

    pool = KVPool(KVPoolConfig(
        num_layers=2, num_blocks=4, block_size=16, num_heads=4, head_dim=64,
        dtype=torch.float16, device=torch.device("cuda"),
    ))
    q = torch.randn(1, 4, 64, dtype=torch.float16, device="cuda")
    bt = torch.tensor([0], dtype=torch.int32, device="cuda")
    with pytest.raises(ValueError):
        paged_decode_attention(q, pool.k_cache, pool.v_cache, bt, 8, layer_idx=99)
