"""
Correctness tests for the Triton FlashAttention forward kernel.

The kernel is validated against two independent references:

  1. A naive materialized PyTorch attention (the pre-FlashAttention
     baseline). This is the "is the math right" check.

  2. ``torch.nn.functional.scaled_dot_product_attention``, which on
     Ampere dispatches to a tuned cuDNN/FA kernel. Matching SDPA within
     fp16 tolerance is the "are we competitive with the reference impl"
     check.

Both references use fp32 softmax internally; we do too. Tolerances are
fp16-realistic: ``atol=5e-3``, ``rtol=5e-2``. Differences smaller than
this come from the order of fp16 accumulations, not from algorithmic
errors.
"""

from __future__ import annotations

import math

import pytest
import torch

cuda_available = torch.cuda.is_available()
pytestmark = pytest.mark.skipif(
    not cuda_available, reason="flash attention tests require CUDA"
)


# ----------------------------------------------------------------------
# References
# ----------------------------------------------------------------------


def _naive_attention(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, causal: bool = True
) -> torch.Tensor:
    """
    Materialized eager attention written in pure PyTorch. Used as the
    primary correctness reference because it has zero algorithmic
    cleverness — every step is exactly what the formula says.

    q: [L_q, H, D]; k, v: [L_k, H, D]. Returns [L_q, H, D].
    Causal mask treats queries as occupying positions [L_k - L_q, L_k).
    """
    L_q, H, D = q.shape
    L_k = k.shape[0]
    scale = 1.0 / math.sqrt(D)

    q_h = q.transpose(0, 1)            # [H, L_q, D]
    k_h = k.transpose(0, 1)            # [H, L_k, D]
    v_h = v.transpose(0, 1)            # [H, L_k, D]

    scores = torch.matmul(q_h, k_h.transpose(-1, -2)) * scale  # [H, L_q, L_k]

    if causal:
        q_pos = torch.arange(L_q, device=q.device) + (L_k - L_q)  # [L_q]
        k_pos = torch.arange(L_k, device=q.device)                 # [L_k]
        mask = k_pos[None, :] > q_pos[:, None]  # [L_q, L_k], True = invalid
        scores = scores.masked_fill(mask.unsqueeze(0), float("-inf"))

    attn = torch.softmax(scores.float(), dim=-1).to(scores.dtype)
    out = torch.matmul(attn, v_h)  # [H, L_q, D]
    return out.transpose(0, 1).contiguous()  # [L_q, H, D]


def _sdpa_attention(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, causal: bool = True
) -> torch.Tensor:
    """``F.scaled_dot_product_attention`` for shapes only equal to L_q == L_k."""
    L_q = q.shape[0]
    L_k = k.shape[0]
    if L_q != L_k:
        raise ValueError("SDPA's is_causal expects square attention; use _naive otherwise")
    q_b = q.transpose(0, 1).unsqueeze(0)  # [1, H, L, D]
    k_b = k.transpose(0, 1).unsqueeze(0)
    v_b = v.transpose(0, 1).unsqueeze(0)
    out = torch.nn.functional.scaled_dot_product_attention(
        q_b, k_b, v_b, is_causal=causal
    )
    return out.squeeze(0).transpose(0, 1).contiguous()


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------


def _random_qkv(L_q: int, L_k: int, H: int, D: int, seed: int = 0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    q = torch.randn(L_q, H, D, dtype=torch.float16, device="cuda", generator=g)
    k = torch.randn(L_k, H, D, dtype=torch.float16, device="cuda", generator=g)
    v = torch.randn(L_k, H, D, dtype=torch.float16, device="cuda", generator=g)
    return q, k, v


# ----------------------------------------------------------------------
# Correctness sweep vs naive
# ----------------------------------------------------------------------


@pytest.mark.parametrize("L", [32, 64, 128, 256, 512, 1024, 2048])
@pytest.mark.parametrize("H", [12])
@pytest.mark.parametrize("D", [64])
def test_triton_matches_naive_self_attention(L: int, H: int, D: int) -> None:
    from src.kernels.triton.flash_attention import triton_flash_attention

    q, k, v = _random_qkv(L, L, H, D, seed=L)
    expected = _naive_attention(q, k, v, causal=True)
    actual = triton_flash_attention(q, k, v, causal=True)
    torch.testing.assert_close(actual, expected, atol=5e-3, rtol=5e-2)


@pytest.mark.parametrize("L", [256, 512, 1024])
def test_triton_matches_sdpa(L: int) -> None:
    from src.kernels.triton.flash_attention import triton_flash_attention

    H, D = 12, 64
    q, k, v = _random_qkv(L, L, H, D, seed=L + 1)
    expected = _sdpa_attention(q, k, v, causal=True)
    actual = triton_flash_attention(q, k, v, causal=True)
    torch.testing.assert_close(actual, expected, atol=5e-3, rtol=5e-2)


# ----------------------------------------------------------------------
# Edge cases
# ----------------------------------------------------------------------


def test_non_aligned_seqlen() -> None:
    """L not divisible by BLOCK_M or BLOCK_N must still produce the right answer."""
    from src.kernels.triton.flash_attention import triton_flash_attention

    L = 100  # not a multiple of 64 or 128
    q, k, v = _random_qkv(L, L, H=4, D=64, seed=7)
    expected = _naive_attention(q, k, v, causal=True)
    actual = triton_flash_attention(q, k, v, causal=True)
    torch.testing.assert_close(actual, expected, atol=5e-3, rtol=5e-2)


def test_single_query_decode_shape() -> None:
    """L_q=1 over a long key cache: the decode-style shape."""
    from src.kernels.triton.flash_attention import triton_flash_attention

    L_k = 97  # deliberately not a multiple of BLOCK_N
    q, k, v = _random_qkv(L_q=1, L_k=L_k, H=12, D=64, seed=11)
    expected = _naive_attention(q, k, v, causal=True)
    actual = triton_flash_attention(q, k, v, causal=True)
    torch.testing.assert_close(actual, expected, atol=5e-3, rtol=5e-2)


def test_short_query_long_cache() -> None:
    """L_q < L_k: prefill of a recompute-preempted sequence."""
    from src.kernels.triton.flash_attention import triton_flash_attention

    q, k, v = _random_qkv(L_q=5, L_k=200, H=8, D=64, seed=13)
    expected = _naive_attention(q, k, v, causal=True)
    actual = triton_flash_attention(q, k, v, causal=True)
    torch.testing.assert_close(actual, expected, atol=5e-3, rtol=5e-2)


def test_head_dim_32() -> None:
    """Smaller head_dim variant."""
    from src.kernels.triton.flash_attention import triton_flash_attention

    q, k, v = _random_qkv(L_q=128, L_k=128, H=8, D=32, seed=17)
    expected = _naive_attention(q, k, v, causal=True)
    actual = triton_flash_attention(q, k, v, causal=True)
    torch.testing.assert_close(actual, expected, atol=5e-3, rtol=5e-2)


def test_head_dim_128() -> None:
    """Larger head_dim variant."""
    from src.kernels.triton.flash_attention import triton_flash_attention

    q, k, v = _random_qkv(L_q=64, L_k=64, H=4, D=128, seed=19)
    expected = _naive_attention(q, k, v, causal=True)
    actual = triton_flash_attention(q, k, v, causal=True)
    torch.testing.assert_close(actual, expected, atol=5e-3, rtol=5e-2)


def test_non_causal_full_attention() -> None:
    """causal=False covers the full key range with no mask."""
    from src.kernels.triton.flash_attention import triton_flash_attention

    q, k, v = _random_qkv(L_q=64, L_k=64, H=4, D=64, seed=23)
    expected = _naive_attention(q, k, v, causal=False)
    actual = triton_flash_attention(q, k, v, causal=False)
    torch.testing.assert_close(actual, expected, atol=5e-3, rtol=5e-2)


# ----------------------------------------------------------------------
# Validation rejections
# ----------------------------------------------------------------------


def test_rejects_fp32() -> None:
    from src.kernels.triton.flash_attention import triton_flash_attention

    q = torch.randn(8, 4, 64, dtype=torch.float32, device="cuda")
    with pytest.raises(TypeError):
        triton_flash_attention(q, q, q)


def test_rejects_cpu_tensor() -> None:
    from src.kernels.triton.flash_attention import triton_flash_attention

    q = torch.randn(8, 4, 64, dtype=torch.float16)
    with pytest.raises(ValueError):
        triton_flash_attention(q, q, q)


def test_rejects_unsupported_head_dim() -> None:
    from src.kernels.triton.flash_attention import triton_flash_attention

    q = torch.randn(8, 4, 48, dtype=torch.float16, device="cuda")
    with pytest.raises(ValueError):
        triton_flash_attention(q, q, q)


def test_rejects_causal_with_lq_gt_lk() -> None:
    from src.kernels.triton.flash_attention import triton_flash_attention

    q = torch.randn(20, 4, 64, dtype=torch.float16, device="cuda")
    k = torch.randn(10, 4, 64, dtype=torch.float16, device="cuda")
    with pytest.raises(ValueError):
        triton_flash_attention(q, k, k, causal=True)
