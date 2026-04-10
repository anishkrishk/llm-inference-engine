"""
FlashAttention forward pass in Triton, for causal self-attention.

FlashAttention is not a different algorithm from regular attention —
it is the same math ``softmax(QK^T / sqrt(d)) V`` laid out in a way
that never materializes the full L x L score matrix. The core trick
is the online softmax update of Milakov & Gimelshein (2018), which
lets us process the keys in tiles and maintain a running output
without needing to see all scores at once.

The loop, per query tile
------------------------
We iterate key tiles ``j = 0, BLOCK_N, 2*BLOCK_N, ...`` and maintain
three pieces of state for each query row in the tile:

    m_i   [BLOCK_M]            running max of scores seen so far
    l_i   [BLOCK_M]            running sum of exp(scores - m_i)
    acc   [BLOCK_M, HEAD_DIM]  running weighted-sum of values

For each new tile of keys we compute

    s        = q @ k.T * scale              # [BLOCK_M, BLOCK_N]
    m_new    = max(m_i, max_row(s))
    alpha    = exp(m_i - m_new)              # fix up the old acc
    p        = exp(s - m_new[:, None])       # tile probs (unnormalized)
    l_new    = alpha * l_i + sum_row(p)
    acc_new  = alpha[:, None] * acc + p @ v

At the end of the loop ``acc / l_i`` is the softmax-weighted output.

Why that works (sketch)
-----------------------
If we had run softmax on the full concatenated score vector ``s_all``,
with global max ``M = max(s_all)``, the output would be

    sum_j exp(s_all_j - M) * v_j
    -----------------------------
        sum_j exp(s_all_j - M)

The online update keeps exactly these two quantities (numerator and
denominator) up to the correction factor ``exp(m_i - m_new)`` that
replaces the "running M" with the "new M". You can verify algebraically
that ``alpha * l_old + sum p_new`` equals the true denominator for the
new max, and similarly for the numerator. See Milakov-Gimelshein 2018
for the derivation; it is three lines of algebra.

Memory footprint
----------------
Naive materialized attention at L=2048, H=12, fp16 holds a
[L, L, H] score tensor plus a [L, L, H] attention tensor — each
about 96 MB, 192 MB combined. This kernel holds only the output
[L, H, d] (~3 MB for the same shape) plus the tiny per-program state,
independent of L.

Supported shapes
----------------
Inputs are ``[L_q, H, D]`` and ``[L_k, H, D]``. HEAD_DIM (``D``) must be
a power of two in [16, 256]. When ``causal=True`` the queries are
treated as occupying the trailing positions of a sequence of length
``L_k``, i.e. query ``i`` sits at absolute position ``(L_k - L_q) + i``.
This means the same kernel handles both prefill (``L_q == L_k``,
queries at 0..L_q-1) and decode (``L_q == 1``, ``L_k == cache_len``).
"""

from __future__ import annotations

import math

import torch
import triton
import triton.language as tl


@triton.jit
def _fwd_kernel(
    Q_ptr, K_ptr, V_ptr, O_ptr,
    stride_qn, stride_qh, stride_qd,
    stride_kn, stride_kh, stride_kd,
    stride_vn, stride_vh, stride_vd,
    stride_on, stride_oh, stride_od,
    N_Q, N_K,
    sm_scale,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
):
    # One program per (query tile, head).
    start_m = tl.program_id(0)
    off_h = tl.program_id(1)

    # Index vectors inside the current program.
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)  # query rows
    offs_n = tl.arange(0, BLOCK_N)                      # key rows within a tile
    offs_d = tl.arange(0, HEAD_DIM)

    # --- Load the Q tile once and keep it in registers for the whole loop.
    q_ptrs = Q_ptr + (
        offs_m[:, None] * stride_qn
        + off_h * stride_qh
        + offs_d[None, :] * stride_qd
    )
    q_mask = offs_m[:, None] < N_Q
    q = tl.load(q_ptrs, mask=q_mask, other=0.0)  # [BLOCK_M, HEAD_DIM], fp16

    # --- Online softmax state. fp32 for numerical stability.
    m_i = tl.full((BLOCK_M,), value=-float("inf"), dtype=tl.float32)
    l_i = tl.zeros((BLOCK_M,), dtype=tl.float32)
    acc = tl.zeros((BLOCK_M, HEAD_DIM), dtype=tl.float32)

    # --- Causal attention only touches keys up to the current query's
    #     absolute position. For causal, the highest query position in
    #     this tile is (start_m + 1) * BLOCK_M - 1 + (N_K - N_Q), so we
    #     can clip the key loop upper bound accordingly. For non-causal
    #     we read the full N_K.
    if IS_CAUSAL:
        hi = tl.minimum((start_m + 1) * BLOCK_M + (N_K - N_Q), N_K)
    else:
        hi = N_K

    # --- Key/value tile loop.
    for start_n in range(0, hi, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        k_positions = start_n + offs_n  # absolute key positions in this tile

        # Load K and V tiles.
        k_bounds = k_positions[:, None] < N_K
        k_ptrs = K_ptr + (
            k_positions[:, None] * stride_kn
            + off_h * stride_kh
            + offs_d[None, :] * stride_kd
        )
        k = tl.load(k_ptrs, mask=k_bounds, other=0.0)  # [BLOCK_N, HEAD_DIM]

        v_ptrs = V_ptr + (
            k_positions[:, None] * stride_vn
            + off_h * stride_vh
            + offs_d[None, :] * stride_vd
        )
        v = tl.load(v_ptrs, mask=k_bounds, other=0.0)  # [BLOCK_N, HEAD_DIM]

        # Scores: [BLOCK_M, BLOCK_N] in fp32 (tl.dot accumulates in fp32).
        s = tl.dot(q, tl.trans(k)) * sm_scale

        # Kill out-of-bounds keys.
        s = tl.where(k_positions[None, :] < N_K, s, -float("inf"))

        # Apply causal mask. Query at row i in this tile has absolute
        # position offs_m[i] + (N_K - N_Q); it can only attend to keys
        # at positions <= its own absolute position.
        if IS_CAUSAL:
            q_abs = offs_m + (N_K - N_Q)  # [BLOCK_M]
            causal_ok = q_abs[:, None] >= k_positions[None, :]
            s = tl.where(causal_ok, s, -float("inf"))

        # Online softmax update.
        m_new = tl.maximum(m_i, tl.max(s, axis=1))   # [BLOCK_M]
        alpha = tl.exp(m_i - m_new)                  # [BLOCK_M]
        p = tl.exp(s - m_new[:, None])               # [BLOCK_M, BLOCK_N]

        # Correct the existing accumulator and fold in this tile.
        acc = acc * alpha[:, None]
        acc = acc + tl.dot(p.to(v.dtype), v)

        # Update running normalizer and max.
        l_i = l_i * alpha + tl.sum(p, axis=1)
        m_i = m_new

    # Normalize the accumulator by the running softmax denominator.
    # Guard against l_i == 0 (only possible for fully-masked rows that
    # do not receive any score updates; they are out-of-bounds queries
    # masked at store time, so the value does not matter, but we still
    # need to avoid a NaN propagating into tl.store).
    l_safe = tl.where(l_i == 0.0, 1.0, l_i)
    acc = acc / l_safe[:, None]

    # Write the output tile.
    o_ptrs = O_ptr + (
        offs_m[:, None] * stride_on
        + off_h * stride_oh
        + offs_d[None, :] * stride_od
    )
    tl.store(o_ptrs, acc.to(O_ptr.dtype.element_ty), mask=offs_m[:, None] < N_Q)


# ---- Python wrapper --------------------------------------------------


def triton_flash_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    causal: bool = True,
) -> torch.Tensor:
    """
    Compute softmax(QK^T / sqrt(d)) V using the Triton kernel above.

    Arguments
    ---------
    q      : [L_q, H, D] fp16, CUDA
    k      : [L_k, H, D] fp16, CUDA
    v      : [L_k, H, D] fp16, CUDA
    causal : if True, queries are treated as occupying trailing
             positions [L_k - L_q, L_k) and a lower-triangular mask
             is applied.

    Returns
    -------
    out : [L_q, H, D] fp16, CUDA
    """
    if not (q.is_cuda and k.is_cuda and v.is_cuda):
        raise ValueError("triton_flash_attention requires CUDA tensors")
    if q.dtype != torch.float16 or k.dtype != torch.float16 or v.dtype != torch.float16:
        raise TypeError(
            f"triton_flash_attention currently supports fp16 only; got "
            f"{q.dtype}, {k.dtype}, {v.dtype}"
        )
    if q.dim() != 3 or k.dim() != 3 or v.dim() != 3:
        raise ValueError(
            f"expected 3D tensors [L, H, D]; got shapes "
            f"{tuple(q.shape)}, {tuple(k.shape)}, {tuple(v.shape)}"
        )
    if q.shape[1:] != k.shape[1:] or k.shape != v.shape:
        raise ValueError(
            f"shape mismatch: q={tuple(q.shape)} k={tuple(k.shape)} v={tuple(v.shape)}"
        )

    L_q, H, D = q.shape
    L_k = k.shape[0]
    if L_q == 0 or L_k == 0:
        raise ValueError(f"zero-length sequence: L_q={L_q}, L_k={L_k}")
    if causal and L_q > L_k:
        raise ValueError(
            f"causal attention requires L_q <= L_k (queries are the "
            f"trailing positions); got L_q={L_q} > L_k={L_k}"
        )
    if D not in (16, 32, 64, 128, 256):
        raise ValueError(
            f"head_dim must be a power of two in [16, 256]; got {D}"
        )

    # Triton strides are in elements, which is what tensor.stride() returns.
    q_c = q.contiguous() if not q.is_contiguous() else q
    k_c = k.contiguous() if not k.is_contiguous() else k
    v_c = v.contiguous() if not v.is_contiguous() else v
    out = torch.empty_like(q_c)

    BLOCK_M = 128
    BLOCK_N = 64

    grid = (triton.cdiv(L_q, BLOCK_M), H)
    sm_scale = 1.0 / math.sqrt(D)

    _fwd_kernel[grid](
        q_c, k_c, v_c, out,
        q_c.stride(0), q_c.stride(1), q_c.stride(2),
        k_c.stride(0), k_c.stride(1), k_c.stride(2),
        v_c.stride(0), v_c.stride(1), v_c.stride(2),
        out.stride(0), out.stride(1), out.stride(2),
        N_Q=L_q, N_K=L_k,
        sm_scale=sm_scale,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        HEAD_DIM=D,
        IS_CAUSAL=causal,
    )
    return out
