"""
Microbenchmark comparing three causal-attention forward implementations
at GPT-2 small shapes (H=12, D=64, fp16) on CUDA.

Implementations
---------------
naive  : pure-PyTorch materialized attention. Builds the full L x L
         score and attention tensors. The "before FlashAttention" baseline.

sdpa   : torch.nn.functional.scaled_dot_product_attention. On Ampere
         this dispatches to a tuned cuDNN/FA kernel; it is what a user
         calling PyTorch's blessed API would get.

triton : our hand-written Triton FlashAttention kernel.

Run from repo root:
    python benchmarks/bench_attention.py
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


def naive_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Materialized causal attention; the pre-FlashAttention baseline."""
    L, H, D = q.shape
    scale = 1.0 / math.sqrt(D)
    q_h = q.transpose(0, 1)
    k_h = k.transpose(0, 1)
    v_h = v.transpose(0, 1)
    scores = torch.matmul(q_h, k_h.transpose(-1, -2)) * scale
    mask = torch.triu(torch.ones(L, L, device=q.device, dtype=torch.bool), diagonal=1)
    scores = scores.masked_fill(mask, float("-inf"))
    attn = torch.softmax(scores.float(), dim=-1).to(scores.dtype)
    return torch.matmul(attn, v_h).transpose(0, 1).contiguous()


def sdpa_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    q_b = q.transpose(0, 1).unsqueeze(0)
    k_b = k.transpose(0, 1).unsqueeze(0)
    v_b = v.transpose(0, 1).unsqueeze(0)
    out = F.scaled_dot_product_attention(q_b, k_b, v_b, is_causal=True)
    return out.squeeze(0).transpose(0, 1).contiguous()


def triton_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    return triton_flash_attention(q, k, v, causal=True)


def bench(fn, q, k, v, n_warmup: int = 10, n_runs: int = 50) -> float:
    """Returns ms per call, median over `n_runs`."""
    for _ in range(n_warmup):
        fn(q, k, v)
    torch.cuda.synchronize()

    timings = []
    for _ in range(n_runs):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn(q, k, v)
        end.record()
        torch.cuda.synchronize()
        timings.append(start.elapsed_time(end))
    timings.sort()
    return timings[len(timings) // 2]


def causal_flops(L: int, H: int, D: int) -> float:
    """
    Useful FLOPs for one causal attention forward.

      QK^T :  H * (L * (L+1)/2) * D * 2  (only the lower triangle)
      PV   :  same shape, same FLOP count
      Total ~ 2 * H * L^2 * D  (we ignore the /2 since softmax adds a few %)
    """
    return 2 * 2 * H * L * L * D


def main() -> None:
    if not torch.cuda.is_available():
        print("CUDA not available; this benchmark requires a GPU.", file=sys.stderr)
        sys.exit(1)

    H, D = 12, 64
    dtype = torch.float16
    device = torch.device("cuda")
    seqlens = [128, 512, 1024, 2048]

    print(
        f"\nattention microbench  |  H={H}, D={D}, fp16, causal  |  "
        f"GPU: {torch.cuda.get_device_name(device)}\n"
    )
    header = (
        f"{'seqlen':>8} | {'naive ms':>10} | {'sdpa ms':>10} | "
        f"{'triton ms':>10} | {'triton tflops':>14} | "
        f"{'triton vs naive':>16} | {'triton vs sdpa':>16}"
    )
    print(header)
    print("-" * len(header))

    for L in seqlens:
        q = torch.randn(L, H, D, dtype=dtype, device=device)
        k = torch.randn(L, H, D, dtype=dtype, device=device)
        v = torch.randn(L, H, D, dtype=dtype, device=device)

        try:
            t_naive = bench(naive_attention, q, k, v)
        except torch.cuda.OutOfMemoryError:
            t_naive = float("nan")
            torch.cuda.empty_cache()

        try:
            t_sdpa = bench(sdpa_attention, q, k, v)
        except torch.cuda.OutOfMemoryError:
            t_sdpa = float("nan")
            torch.cuda.empty_cache()

        t_triton = bench(triton_attention, q, k, v)

        flops = causal_flops(L, H, D)
        triton_tflops = (flops / (t_triton * 1e-3)) / 1e12

        speedup_naive = t_naive / t_triton if not math.isnan(t_naive) else float("nan")
        speedup_sdpa = t_sdpa / t_triton if not math.isnan(t_sdpa) else float("nan")

        def fmt(x: float) -> str:
            return "OOM" if math.isnan(x) else f"{x:>10.3f}"

        def fmt_speedup(x: float) -> str:
            return "  n/a" if math.isnan(x) else f"{x:>14.2f}x"

        print(
            f"{L:>8} | {fmt(t_naive)} | {fmt(t_sdpa)} | {t_triton:>10.3f} "
            f"| {triton_tflops:>14.2f} | {fmt_speedup(speedup_naive):>16} "
            f"| {fmt_speedup(speedup_sdpa):>16}"
        )

    print()


if __name__ == "__main__":
    main()
