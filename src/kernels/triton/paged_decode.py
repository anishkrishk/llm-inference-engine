"""
Paged decode-attention forward kernel.

This kernel is the partner to the FlashAttention prefill kernel in
``flash_attention.py``. Where the prefill kernel handles many-query
self-attention on contiguous K/V tensors, this kernel handles the
single-query case where K and V live in non-contiguous physical
blocks indexed via a sequence's block table.

Why a separate kernel
---------------------
At decode time we have:

    Q : [num_heads, head_dim]                  (one new token's projection)
    K : scattered across {block_table[0], block_table[1], ...} blocks
    V : scattered across the same blocks

The "obvious" approach is to gather K and V into contiguous tensors
first via ``KVPool.read()`` and then call any normal attention kernel.
The gather works but it costs ``num_layers * 2 * context_len * H * D``
of memory traffic per token, which on a 3060 (~360 GB/s) becomes the
dominant cost at long contexts.

This kernel skips the gather entirely. It loops over the block-table
entries, loads one block of K and V at a time directly from the pool's
backing tensors using the *physical* block id, and folds each block
into a running online-softmax accumulator. The intermediate gathered
tensor never exists.

The math is identical to ``flash_attention.py``. The differences are
all in the access pattern:

  * ``BLOCK_M = 1``: a single query row.
  * ``BLOCK_N = block_size`` (16 for our pool): the unit of K/V layout
    is the physical block, so there's no reason to use a different
    tile size — and this lines up the kernel's tile boundaries with
    the cache's storage boundaries for free.
  * The K and V pointers are recomputed each iteration from the
    block-table indirection: ``physical = tl.load(block_table + i)``,
    then ``k_ptr = K_cache + layer*L_stride + physical*B_stride + ...``.
  * No ``tl.dot``: with one query row there's no tensor-core matmul to
    issue (tensor cores want M >= 16), so we use elementwise multiply
    + reduction. At decode the workload is memory-bound anyway, so
    losing tensor-core throughput is irrelevant.

Triton vs CUDA for this kernel
------------------------------
vLLM v1 wrote paged attention in Triton; v2 rewrote it in CUDA for
finer-grained shared-memory pipelining at very long contexts. For
GPT-2 small on a 3060 the kernel is memory-bandwidth-limited and the
two implementations would hit the same ceiling. We use Triton: same
correctness, ~1/5 the lines of code, no ``setup.py`` extension build
to maintain. The cases where CUDA is meaningfully faster are at
context lengths >10K and on hardware with more aggressive memory
subsystems than the 3060.
"""

from __future__ import annotations

import math

import torch
import triton
import triton.language as tl


@triton.jit
def _paged_decode_kernel(
    Q_ptr,                  # [H, D] — single-query Q for one sequence
    K_cache_ptr,            # [L_layers, num_blocks, block_size, H, D]
    V_cache_ptr,            # same shape
    Out_ptr,                # [H, D]
    block_table_ptr,        # int32, [num_blocks_for_seq]
    context_len,            # i32 — number of valid cached positions
    layer_idx,              # i32
    stride_q_h, stride_q_d,
    stride_k_layer, stride_k_block, stride_k_slot, stride_k_head, stride_k_d,
    stride_v_layer, stride_v_block, stride_v_slot, stride_v_head, stride_v_d,
    stride_o_h, stride_o_d,
    sm_scale,
    BLOCK_SIZE: tl.constexpr,
    HEAD_DIM: tl.constexpr,
):
    # One program per head. Grid = (num_heads,).
    off_h = tl.program_id(0)

    offs_d = tl.arange(0, HEAD_DIM)
    offs_slot = tl.arange(0, BLOCK_SIZE)

    # Load Q for this head, upcast to fp32 once for repeated use.
    q_ptrs = Q_ptr + off_h * stride_q_h + offs_d * stride_q_d
    q = tl.load(q_ptrs).to(tl.float32)  # [HEAD_DIM] fp32

    # Online softmax state. m_i and l_i are tracked as scalars; acc is
    # the running output vector for this head, fp32 throughout.
    m_i = -float("inf")
    l_i = 0.0
    acc = tl.zeros((HEAD_DIM,), dtype=tl.float32)

    # Iterate over the block-table entries that cover the valid context.
    num_blocks_used = tl.cdiv(context_len, BLOCK_SIZE)

    for block_idx in range(0, num_blocks_used):
        # The indirection — read this sequence's physical block id from
        # the block table on the fly.
        physical_block = tl.load(block_table_ptr + block_idx).to(tl.int64)

        # Compute K and V tile pointers into the layer's slab. The
        # pointer math is the same as for any other 5-D tensor index;
        # the only nontrivial term is the dynamic ``physical_block``
        # we just loaded.
        k_ptrs = (
            K_cache_ptr
            + layer_idx * stride_k_layer
            + physical_block * stride_k_block
            + offs_slot[:, None] * stride_k_slot
            + off_h * stride_k_head
            + offs_d[None, :] * stride_k_d
        )
        v_ptrs = (
            V_cache_ptr
            + layer_idx * stride_v_layer
            + physical_block * stride_v_block
            + offs_slot[:, None] * stride_v_slot
            + off_h * stride_v_head
            + offs_d[None, :] * stride_v_d
        )

        # Last block may be only partially populated; mask the unused slots.
        block_start = block_idx * BLOCK_SIZE
        positions = block_start + offs_slot                # [BLOCK_SIZE]
        valid = positions < context_len                    # [BLOCK_SIZE]

        k = tl.load(k_ptrs, mask=valid[:, None], other=0.0).to(tl.float32)
        v = tl.load(v_ptrs, mask=valid[:, None], other=0.0).to(tl.float32)

        # Scores for this tile: for each cached position i in the tile,
        # score[i] = sum over d of q[d] * k[i, d], scaled by 1/sqrt(d).
        # Done as elementwise multiply + reduction because BLOCK_M=1.
        scores = tl.sum(q[None, :] * k, axis=1) * sm_scale  # [BLOCK_SIZE]
        scores = tl.where(valid, scores, -float("inf"))

        # Online softmax update — same form as the prefill kernel,
        # but with scalar m_i, l_i since we have only one query row.
        m_block = tl.max(scores, axis=0)
        m_new = tl.maximum(m_i, m_block)
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(scores - m_new)  # [BLOCK_SIZE]

        # Fold the tile's contribution into the running accumulator.
        # acc[d] = acc[d] * alpha + sum over i of p[i] * v[i, d]
        acc = acc * alpha + tl.sum(p[:, None] * v, axis=0)
        l_i = l_i * alpha + tl.sum(p, axis=0)
        m_i = m_new

    # Normalize by the accumulated softmax denominator and write.
    out = acc / l_i

    o_ptrs = Out_ptr + off_h * stride_o_h + offs_d * stride_o_d
    tl.store(o_ptrs, out.to(Out_ptr.dtype.element_ty))


@triton.jit
def _batched_paged_decode_kernel(
    Q_ptr,
    K_cache_ptr, V_cache_ptr,
    Out_ptr,
    block_tables_ptr,
    context_lens_ptr,
    stride_q_b, stride_q_h, stride_q_d,
    stride_k_layer, stride_k_block, stride_k_slot, stride_k_head, stride_k_d,
    stride_v_layer, stride_v_block, stride_v_slot, stride_v_head, stride_v_d,
    stride_o_b, stride_o_h, stride_o_d,
    stride_bt_b, stride_bt_n,
    layer_idx,
    sm_scale,
    BLOCK_SIZE: tl.constexpr,
    HEAD_DIM: tl.constexpr,
):
    """One program per (head, sequence). Grid = (num_heads, batch_size)."""
    off_h = tl.program_id(0)
    off_b = tl.program_id(1)

    context_len = tl.load(context_lens_ptr + off_b).to(tl.int32)

    offs_d = tl.arange(0, HEAD_DIM)
    offs_slot = tl.arange(0, BLOCK_SIZE)

    q_ptrs = Q_ptr + off_b * stride_q_b + off_h * stride_q_h + offs_d * stride_q_d
    q = tl.load(q_ptrs).to(tl.float32)

    m_i = -float("inf")
    l_i = 0.0
    acc = tl.zeros((HEAD_DIM,), dtype=tl.float32)

    num_blocks_used = tl.cdiv(context_len, BLOCK_SIZE)

    for block_idx in range(0, num_blocks_used):
        physical_block = tl.load(
            block_tables_ptr + off_b * stride_bt_b + block_idx * stride_bt_n
        ).to(tl.int64)

        k_ptrs = (
            K_cache_ptr
            + layer_idx * stride_k_layer
            + physical_block * stride_k_block
            + offs_slot[:, None] * stride_k_slot
            + off_h * stride_k_head
            + offs_d[None, :] * stride_k_d
        )
        v_ptrs = (
            V_cache_ptr
            + layer_idx * stride_v_layer
            + physical_block * stride_v_block
            + offs_slot[:, None] * stride_v_slot
            + off_h * stride_v_head
            + offs_d[None, :] * stride_v_d
        )

        block_start = block_idx * BLOCK_SIZE
        positions = block_start + offs_slot
        valid = positions < context_len

        k = tl.load(k_ptrs, mask=valid[:, None], other=0.0).to(tl.float32)
        v = tl.load(v_ptrs, mask=valid[:, None], other=0.0).to(tl.float32)

        scores = tl.sum(q[None, :] * k, axis=1) * sm_scale
        scores = tl.where(valid, scores, -float("inf"))

        m_block = tl.max(scores, axis=0)
        m_new = tl.maximum(m_i, m_block)
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(scores - m_new)

        acc = acc * alpha + tl.sum(p[:, None] * v, axis=0)
        l_i = l_i * alpha + tl.sum(p, axis=0)
        m_i = m_new

    out = acc / l_i
    o_ptrs = Out_ptr + off_b * stride_o_b + off_h * stride_o_h + offs_d * stride_o_d
    tl.store(o_ptrs, out.to(Out_ptr.dtype.element_ty))


def batched_paged_decode_attention(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    block_tables: torch.Tensor,
    context_lens: torch.Tensor,
    layer_idx: int,
) -> torch.Tensor:
    """
    Batched version: process B sequences in one kernel launch.

    Arguments
    ---------
    q            : [B, H, D] fp16, CUDA
    k_cache      : [num_layers, num_blocks, block_size, H, D] fp16
    v_cache      : same
    block_tables : [B, max_num_blocks] int32, CUDA — padded to max length
    context_lens : [B] int32, CUDA — per-sequence context lengths
    layer_idx    : int

    Returns
    -------
    out : [B, H, D] fp16
    """
    B, H, D = q.shape
    block_size = k_cache.shape[2]
    out = torch.empty_like(q)
    sm_scale = 1.0 / math.sqrt(D)

    grid = (H, B)

    _batched_paged_decode_kernel[grid](
        q, k_cache, v_cache, out, block_tables, context_lens,
        q.stride(0), q.stride(1), q.stride(2),
        k_cache.stride(0), k_cache.stride(1), k_cache.stride(2),
        k_cache.stride(3), k_cache.stride(4),
        v_cache.stride(0), v_cache.stride(1), v_cache.stride(2),
        v_cache.stride(3), v_cache.stride(4),
        out.stride(0), out.stride(1), out.stride(2),
        block_tables.stride(0), block_tables.stride(1),
        layer_idx,
        sm_scale=sm_scale,
        BLOCK_SIZE=block_size,
        HEAD_DIM=D,
    )
    return out


def paged_decode_attention(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    block_table: torch.Tensor,
    context_len: int,
    layer_idx: int,
) -> torch.Tensor:
    """
    Single-query attention reading K/V directly from a paged KV pool.

    Arguments
    ---------
    q          : [1, H, D] or [H, D] fp16, CUDA. The single new query.
    k_cache    : [num_layers, num_blocks, block_size, H, D] fp16, CUDA.
    v_cache    : same shape as k_cache.
    block_table: 1-D int32 or int64 tensor on CUDA, length >= num blocks
                 needed to cover ``context_len`` positions.
    context_len: number of valid cached positions to attend over,
                 typically ``start_pos + 1`` from the model's view.
    layer_idx  : which layer's slab to read from.

    Returns
    -------
    out : same rank as ``q`` (3-D if q was 3-D, else 2-D), shape
          ``[1, H, D]`` or ``[H, D]``, fp16 on CUDA.
    """
    if not q.is_cuda or not k_cache.is_cuda or not v_cache.is_cuda or not block_table.is_cuda:
        raise ValueError("paged_decode_attention requires CUDA tensors")
    if q.dtype != torch.float16 or k_cache.dtype != torch.float16 or v_cache.dtype != torch.float16:
        raise TypeError(
            f"paged_decode_attention currently supports fp16 only; "
            f"got q={q.dtype}, k={k_cache.dtype}, v={v_cache.dtype}"
        )

    if q.dim() == 3:
        if q.shape[0] != 1:
            raise ValueError(
                f"paged_decode_attention expects exactly one query token, "
                f"got q.shape[0]={q.shape[0]}"
            )
        q_2d = q.squeeze(0)
        wrap_back_to_3d = True
    elif q.dim() == 2:
        q_2d = q
        wrap_back_to_3d = False
    else:
        raise ValueError(f"q must be [1, H, D] or [H, D], got shape {tuple(q.shape)}")

    H, D = q_2d.shape
    if k_cache.dim() != 5:
        raise ValueError(f"k_cache must be 5-D, got shape {tuple(k_cache.shape)}")
    num_layers, num_blocks, block_size, cache_h, cache_d = k_cache.shape
    if (cache_h, cache_d) != (H, D):
        raise ValueError(
            f"q [H={H}, D={D}] does not match cache [H={cache_h}, D={cache_d}]"
        )
    if k_cache.shape != v_cache.shape:
        raise ValueError(
            f"k_cache and v_cache shapes differ: {tuple(k_cache.shape)} vs {tuple(v_cache.shape)}"
        )
    if not 0 <= layer_idx < num_layers:
        raise ValueError(f"layer_idx={layer_idx} out of range [0, {num_layers})")
    if context_len <= 0:
        raise ValueError(f"context_len must be positive, got {context_len}")
    needed_blocks = (context_len + block_size - 1) // block_size
    if block_table.dim() != 1 or block_table.shape[0] < needed_blocks:
        raise ValueError(
            f"block_table must be 1-D with at least {needed_blocks} entries "
            f"to cover context_len={context_len} at block_size={block_size}; "
            f"got shape {tuple(block_table.shape)}"
        )
    if block_table.dtype not in (torch.int32, torch.int64):
        raise TypeError(
            f"block_table must be int32 or int64, got {block_table.dtype}"
        )
    if D not in (16, 32, 64, 128, 256):
        raise ValueError(f"head_dim must be a power of two in [16, 256]; got {D}")

    out = torch.empty_like(q_2d)

    sm_scale = 1.0 / math.sqrt(D)
    grid = (H,)

    _paged_decode_kernel[grid](
        q_2d, k_cache, v_cache, out, block_table,
        context_len, layer_idx,
        q_2d.stride(0), q_2d.stride(1),
        k_cache.stride(0), k_cache.stride(1), k_cache.stride(2),
        k_cache.stride(3), k_cache.stride(4),
        v_cache.stride(0), v_cache.stride(1), v_cache.stride(2),
        v_cache.stride(3), v_cache.stride(4),
        out.stride(0), out.stride(1),
        sm_scale=sm_scale,
        BLOCK_SIZE=block_size,
        HEAD_DIM=D,
    )

    return out.unsqueeze(0) if wrap_back_to_3d else out
