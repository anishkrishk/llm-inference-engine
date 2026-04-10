"""
Microbenchmark for the decode-attention path at varying context lengths.

Three implementations side-by-side, each computing the same single-query
attention output:

  eager-gather  : KVPool.read() + materialized softmax matmul
  triton-gather : KVPool.read() + Triton FlashAttention forward kernel
  paged-direct  : paged_decode_attention (no gather; kernel walks the
                  block table itself)

The point is to make the cost of the gather visible. At short contexts
the gather is essentially free; at long contexts it dominates.

Run from repo root:
    python benchmarks/bench_paged_decode.py
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.kernels.triton.flash_attention import triton_flash_attention  # noqa: E402
from src.kernels.triton.paged_decode import paged_decode_attention  # noqa: E402
from src.kv_cache.kv_pool import KVPool, KVPoolConfig  # noqa: E402


def eager_gather_attention(
    q: torch.Tensor,
    pool: KVPool,
    block_ids: list[int],
    context_len: int,
    layer_idx: int,
) -> torch.Tensor:
    """Reference path: gather K,V then run a materialized softmax matmul."""
    k_cache, v_cache = pool.read(layer_idx, block_ids, context_len)  # [ctx, H, D]
    q_h = q.transpose(0, 1)              # [H, 1, D]
    k_h = k_cache.transpose(0, 1)        # [H, ctx, D]
    v_h = v_cache.transpose(0, 1)        # [H, ctx, D]
    scale = 1.0 / math.sqrt(q.shape[-1])
    scores = torch.matmul(q_h, k_h.transpose(-1, -2)) * scale
    attn = torch.softmax(scores.float(), dim=-1).to(scores.dtype)
    out = torch.matmul(attn, v_h)        # [H, 1, D]
    return out.transpose(0, 1)            # [1, H, D]


def triton_gather_attention(
    q: torch.Tensor,
    pool: KVPool,
    block_ids: list[int],
    context_len: int,
    layer_idx: int,
) -> torch.Tensor:
    """Gather K,V from pool then run the FlashAttention forward kernel."""
    k_cache, v_cache = pool.read(layer_idx, block_ids, context_len)  # [ctx, H, D]
    return triton_flash_attention(q, k_cache, v_cache, causal=True)


def paged_direct_attention(
    q: torch.Tensor,
    pool: KVPool,
    block_table: torch.Tensor,
    context_len: int,
    layer_idx: int,
) -> torch.Tensor:
    """Skip the gather; kernel walks the block table directly."""
    return paged_decode_attention(
        q, pool.k_cache, pool.v_cache, block_table, context_len, layer_idx
    )


def bench(fn, n_warmup: int = 20, n_runs: int = 100) -> float:
    for _ in range(n_warmup):
        fn()
    torch.cuda.synchronize()
    timings = []
    for _ in range(n_runs):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        timings.append(start.elapsed_time(end))
    timings.sort()
    return timings[len(timings) // 2]


def main() -> None:
    if not torch.cuda.is_available():
        print("CUDA not available; this benchmark requires a GPU.", file=sys.stderr)
        sys.exit(1)

    device = torch.device("cuda")
    dtype = torch.float16
    H, D = 12, 64
    block_size = 16
    num_layers = 12
    layer_idx = 6  # arbitrary middle layer

    seqlens = [16, 64, 256, 512, 1024]

    # Allocate a pool large enough for the longest test.
    max_blocks_needed = (max(seqlens) + block_size - 1) // block_size
    pool = KVPool(
        KVPoolConfig(
            num_layers=num_layers,
            num_blocks=max_blocks_needed * 2,
            block_size=block_size,
            num_heads=H,
            head_dim=D,
            dtype=dtype,
            device=device,
        )
    )

    print(
        f"\npaged decode microbench  |  H={H}, D={D}, fp16, GPT-2 small  |  "
        f"{torch.cuda.get_device_name(device)}\n"
    )
    header = (
        f"{'context':>8} | {'eager+gather':>14} | {'triton+gather':>14} "
        f"| {'paged-direct':>13} | {'speedup vs eager':>17} | {'speedup vs triton':>18}"
    )
    print(header)
    print("-" * len(header))

    for ctx in seqlens:
        # Pick a non-contiguous block table to mirror real preempt-and-recycle
        # block id distributions; the kernel must handle this correctly.
        n_blocks = (ctx + block_size - 1) // block_size
        block_ids = list(range(0, 2 * n_blocks, 2))[:n_blocks]  # [0, 2, 4, ...]

        # Populate the cache once.
        for layer in range(num_layers):
            kd = torch.randn(ctx, H, D, dtype=dtype, device=device)
            vd = torch.randn(ctx, H, D, dtype=dtype, device=device)
            pool.write(layer, block_ids, start_pos=0, k_new=kd, v_new=vd)

        q = torch.randn(1, H, D, dtype=dtype, device=device)
        bt_tensor = torch.tensor(block_ids, dtype=torch.int32, device=device)

        t_eager = bench(lambda: eager_gather_attention(q, pool, block_ids, ctx, layer_idx))
        t_triton = bench(lambda: triton_gather_attention(q, pool, block_ids, ctx, layer_idx))
        t_paged = bench(lambda: paged_direct_attention(q, pool, bt_tensor, ctx, layer_idx))

        speedup_eager = t_eager / t_paged
        speedup_triton = t_triton / t_paged

        print(
            f"{ctx:>8} | {t_eager:>11.4f} ms | {t_triton:>11.4f} ms | "
            f"{t_paged:>10.4f} ms | {speedup_eager:>15.2f}x | {speedup_triton:>16.2f}x"
        )

    print()


if __name__ == "__main__":
    main()
