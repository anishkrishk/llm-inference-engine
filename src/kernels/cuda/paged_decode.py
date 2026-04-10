"""
Python wrapper for the CUDA paged decode-attention kernel.

Uses ``torch.utils.cpp_extension.load()`` to JIT-compile the ``.cu``
file on first import. The compiled module is cached in a build
directory so subsequent imports are instant.

The wrapper provides the same signature as the Triton version in
``src.kernels.triton.paged_decode.paged_decode_attention`` so the
two can be compared directly in tests and benchmarks.
"""

from __future__ import annotations

import math
from pathlib import Path

import torch
from torch.utils.cpp_extension import load

# JIT-compile the CUDA kernel. This takes ~30 seconds on first import,
# then is cached for subsequent imports within the same build directory.
_CU_SRC = Path(__file__).parent / "paged_decode.cu"

_module = None


def _get_module():
    global _module
    if _module is None:
        _module = load(
            name="paged_decode_cuda",
            sources=[str(_CU_SRC)],
            verbose=False,
        )
    return _module


def cuda_paged_decode_attention(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    block_table: torch.Tensor,
    context_len: int,
    layer_idx: int,
) -> torch.Tensor:
    """
    Single-query paged decode attention via the CUDA C kernel.

    Arguments
    ---------
    q          : [1, H, D] or [H, D] fp16, CUDA
    k_cache    : [num_layers, num_blocks, block_size, H, D] fp16
    v_cache    : same
    block_table: 1-D int32 CUDA tensor
    context_len: number of valid cached positions
    layer_idx  : which layer's slab

    Returns
    -------
    out : same rank as q, fp16
    """
    if q.dim() == 3:
        if q.shape[0] != 1:
            raise ValueError(f"q must have L=1, got {q.shape[0]}")
        q_2d = q.squeeze(0)
        wrap = True
    elif q.dim() == 2:
        q_2d = q
        wrap = False
    else:
        raise ValueError(f"q must be [1, H, D] or [H, D], got {tuple(q.shape)}")

    mod = _get_module()
    out = mod.paged_decode_attention_cuda(
        q_2d.contiguous(),
        k_cache.contiguous(),
        v_cache.contiguous(),
        block_table.contiguous().to(torch.int32),
        context_len,
        layer_idx,
    )
    return out.unsqueeze(0) if wrap else out
