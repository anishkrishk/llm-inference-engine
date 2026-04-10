"""
Comprehensive benchmark suite for the LLM inference engine.

Runs three benchmark groups and produces matplotlib charts saved to
results/. This is the script that backs every number on the resume
with reproducible evidence.

Groups:
  1. Attention kernel microbench — naive vs SDPA vs Triton FA at
     seqlens [128, 512, 1024, 2048]
  2. Paged decode microbench — eager+gather vs triton+gather vs
     paged-direct at context lengths [16, 64, 256, 512, 1024]
  3. End-to-end throughput — our engine vs HF baseline on a
     20-request mixed-length workload

Charts:
  results/attention_speedup.png
  results/paged_decode_speedup.png
  results/throughput_comparison.png

Run from repo root:
    python benchmarks/run_all.py
"""

from __future__ import annotations

import json
import math
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402


RESULTS_DIR = _REPO_ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)

DEVICE = torch.device("cuda")
DTYPE = torch.float16
H, D = 12, 64  # GPT-2 small


# ======================================================================
# Timing helpers
# ======================================================================


def bench_fn(fn, n_warmup: int = 10, n_runs: int = 50) -> float:
    """Returns median ms per call."""
    for _ in range(n_warmup):
        fn()
    torch.cuda.synchronize()
    timings = []
    for _ in range(n_runs):
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        fn()
        e.record()
        torch.cuda.synchronize()
        timings.append(s.elapsed_time(e))
    timings.sort()
    return timings[len(timings) // 2]


# ======================================================================
# Group 1: Attention kernel microbench
# ======================================================================


def run_attention_bench() -> Dict:
    from src.kernels.triton.flash_attention import triton_flash_attention

    def naive(q, k, v):
        L = q.shape[0]
        sc = 1.0 / math.sqrt(D)
        q_h, k_h, v_h = q.transpose(0, 1), k.transpose(0, 1), v.transpose(0, 1)
        scores = torch.matmul(q_h, k_h.transpose(-1, -2)) * sc
        mask = torch.triu(torch.ones(L, L, device=q.device, dtype=torch.bool), diagonal=1)
        scores = scores.masked_fill(mask, float("-inf"))
        attn = torch.softmax(scores.float(), dim=-1).to(scores.dtype)
        return torch.matmul(attn, v_h).transpose(0, 1).contiguous()

    def sdpa(q, k, v):
        q_b = q.transpose(0, 1).unsqueeze(0)
        k_b = k.transpose(0, 1).unsqueeze(0)
        v_b = v.transpose(0, 1).unsqueeze(0)
        return F.scaled_dot_product_attention(q_b, k_b, v_b, is_causal=True).squeeze(0).transpose(0, 1).contiguous()

    seqlens = [128, 512, 1024, 2048]
    results = {"seqlens": seqlens, "naive_ms": [], "sdpa_ms": [], "triton_ms": []}

    print("\n=== Group 1: Attention kernel microbench ===\n")
    print(f"{'seqlen':>8} | {'naive ms':>10} | {'sdpa ms':>10} | {'triton ms':>10} | {'vs naive':>10} | {'vs sdpa':>10}")
    print("-" * 70)

    for L in seqlens:
        q = torch.randn(L, H, D, dtype=DTYPE, device=DEVICE)
        k = torch.randn(L, H, D, dtype=DTYPE, device=DEVICE)
        v = torch.randn(L, H, D, dtype=DTYPE, device=DEVICE)

        try:
            t_naive = bench_fn(lambda: naive(q, k, v))
        except torch.cuda.OutOfMemoryError:
            t_naive = float("nan")
            torch.cuda.empty_cache()
        t_sdpa = bench_fn(lambda: sdpa(q, k, v))
        t_triton = bench_fn(lambda: triton_flash_attention(q, k, v, causal=True))

        results["naive_ms"].append(t_naive)
        results["sdpa_ms"].append(t_sdpa)
        results["triton_ms"].append(t_triton)

        sp_n = t_naive / t_triton if not math.isnan(t_naive) else float("nan")
        sp_s = t_sdpa / t_triton
        print(f"{L:>8} | {t_naive:>10.3f} | {t_sdpa:>10.3f} | {t_triton:>10.3f} | {sp_n:>9.1f}x | {sp_s:>9.2f}x")

    return results


def plot_attention(data: Dict) -> None:
    seqlens = data["seqlens"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Left: absolute latency
    ax1.plot(seqlens, data["naive_ms"], "o-", label="naive materialized", color="#d62728")
    ax1.plot(seqlens, data["sdpa_ms"], "s-", label="F.sdpa (cuDNN)", color="#2ca02c")
    ax1.plot(seqlens, data["triton_ms"], "^-", label="our Triton FA", color="#1f77b4")
    ax1.set_xlabel("Sequence length")
    ax1.set_ylabel("Latency (ms)")
    ax1.set_title("Causal attention forward (H=12, D=64, fp16)")
    ax1.legend()
    ax1.set_yscale("log")
    ax1.grid(True, alpha=0.3)

    # Right: speedup vs naive
    speedups = [n / t for n, t in zip(data["naive_ms"], data["triton_ms"]) if not math.isnan(n)]
    valid_seqlens = [s for s, n in zip(seqlens, data["naive_ms"]) if not math.isnan(n)]
    ax2.bar(range(len(valid_seqlens)), speedups, color="#1f77b4", alpha=0.8)
    ax2.set_xticks(range(len(valid_seqlens)))
    ax2.set_xticklabels([str(s) for s in valid_seqlens])
    ax2.set_xlabel("Sequence length")
    ax2.set_ylabel("Speedup (x)")
    ax2.set_title("Triton FA speedup vs naive materialized")
    for i, v in enumerate(speedups):
        ax2.text(i, v + 0.3, f"{v:.1f}x", ha="center", fontsize=10)
    ax2.grid(True, alpha=0.3, axis="y")

    fig.tight_layout()
    path = RESULTS_DIR / "attention_speedup.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  saved {path}")


# ======================================================================
# Group 2: Paged decode microbench
# ======================================================================


def run_paged_decode_bench() -> Dict:
    from src.kernels.triton.flash_attention import triton_flash_attention
    from src.kernels.triton.paged_decode import paged_decode_attention
    from src.kv_cache.kv_pool import KVPool, KVPoolConfig

    ctx_lens = [16, 64, 256, 512, 1024]
    block_size = 16
    max_blocks = (max(ctx_lens) + block_size - 1) // block_size
    pool = KVPool(KVPoolConfig(
        num_layers=12, num_blocks=max_blocks * 2, block_size=block_size,
        num_heads=H, head_dim=D, dtype=DTYPE, device=DEVICE,
    ))

    results = {"ctx_lens": ctx_lens, "eager_ms": [], "triton_ms": [], "paged_ms": []}

    print("\n=== Group 2: Paged decode microbench ===\n")
    print(f"{'context':>8} | {'eager+gather':>14} | {'triton+gather':>14} | {'paged-direct':>13} | {'vs eager':>10}")
    print("-" * 70)

    for ctx in ctx_lens:
        n_blocks = (ctx + block_size - 1) // block_size
        block_ids = list(range(0, 2 * n_blocks, 2))[:n_blocks]
        for layer in range(12):
            kd = torch.randn(ctx, H, D, dtype=DTYPE, device=DEVICE)
            vd = torch.randn(ctx, H, D, dtype=DTYPE, device=DEVICE)
            pool.write(layer, block_ids, 0, kd, vd)

        q = torch.randn(1, H, D, dtype=DTYPE, device=DEVICE)
        bt = torch.tensor(block_ids, dtype=torch.int32, device=DEVICE)

        def eager():
            kc, vc = pool.read(6, block_ids, ctx)
            sc = 1.0 / math.sqrt(D)
            q_h = q.transpose(0, 1)
            k_h = kc.transpose(0, 1)
            v_h = vc.transpose(0, 1)
            scores = torch.matmul(q_h, k_h.transpose(-1, -2)) * sc
            return torch.softmax(scores.float(), dim=-1).to(scores.dtype) @ v_h

        def triton_g():
            kc, vc = pool.read(6, block_ids, ctx)
            return triton_flash_attention(q, kc, vc, causal=True)

        def paged():
            return paged_decode_attention(q, pool.k_cache, pool.v_cache, bt, ctx, 6)

        t_e = bench_fn(eager)
        t_t = bench_fn(triton_g)
        t_p = bench_fn(paged)
        results["eager_ms"].append(t_e)
        results["triton_ms"].append(t_t)
        results["paged_ms"].append(t_p)
        print(f"{ctx:>8} | {t_e:>11.4f} ms | {t_t:>11.4f} ms | {t_p:>10.4f} ms | {t_e / t_p:>9.2f}x")

    return results


def plot_paged_decode(data: Dict) -> None:
    ctx_lens = data["ctx_lens"]
    fig, ax = plt.subplots(figsize=(8, 5))
    x = range(len(ctx_lens))
    w = 0.25
    ax.bar([i - w for i in x], data["eager_ms"], w, label="eager + gather", color="#d62728", alpha=0.8)
    ax.bar([i for i in x], data["triton_ms"], w, label="triton FA + gather", color="#2ca02c", alpha=0.8)
    ax.bar([i + w for i in x], data["paged_ms"], w, label="paged-direct (ours)", color="#1f77b4", alpha=0.8)
    ax.set_xticks(list(x))
    ax.set_xticklabels([str(c) for c in ctx_lens])
    ax.set_xlabel("Context length (tokens)")
    ax.set_ylabel("Latency (ms)")
    ax.set_title("Decode attention: per-call latency (H=12, D=64, fp16)")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    path = RESULTS_DIR / "paged_decode_speedup.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  saved {path}")


# ======================================================================
# Group 3: End-to-end throughput
# ======================================================================


@dataclass
class BenchReq:
    rid: str
    prompt_ids: List[int]
    max_new: int


def build_reqs(n: int, seed: int = 42) -> List[BenchReq]:
    rng = random.Random(seed)
    return [
        BenchReq(f"r{i:02d}", list(range(1, rng.randint(16, 256) + 1)), rng.randint(32, 64))
        for i in range(n)
    ]


def run_throughput_bench() -> Dict:
    from transformers import GPT2LMHeadModel
    from src.kv_cache.kv_pool import KVPool, KVPoolConfig
    from src.model.gpt2 import GPT2Config, load_gpt2_from_hf
    from src.model.gpt2_runner import GPT2Runner
    from src.model.types import GenerateInput
    from src.scheduler.engine import Engine, EngineConfig

    reqs = build_reqs(20)
    total_out = sum(r.max_new for r in reqs)
    total_prompt = sum(len(r.prompt_ids) for r in reqs)

    print("\n=== Group 3: End-to-end throughput ===\n")
    print(f"  {len(reqs)} requests, {total_prompt} prompt tokens, {total_out} output tokens")
    prompt_lens = [len(r.prompt_ids) for r in reqs]
    print(f"  prompt range: {min(prompt_lens)}–{max(prompt_lens)} (mean {sum(prompt_lens)/len(prompt_lens):.0f})")

    # HF baseline
    print("  running HF baseline...")
    hf = GPT2LMHeadModel.from_pretrained("gpt2").to(device=DEVICE, dtype=DTYPE).eval()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.inference_mode():
        for r in reqs:
            ids = torch.tensor([r.prompt_ids], device=DEVICE)
            hf.generate(ids, max_new_tokens=r.max_new, do_sample=False, pad_token_id=50256)
    torch.cuda.synchronize()
    hf_time = time.perf_counter() - t0
    del hf
    torch.cuda.empty_cache()

    # Our engine
    print("  running our engine...")
    gpt2_cfg = GPT2Config.gpt2_small()
    pool = KVPool(KVPoolConfig(
        num_layers=12, num_blocks=256, block_size=16,
        num_heads=12, head_dim=64, dtype=DTYPE, device=DEVICE,
    ))
    model = load_gpt2_from_hf("gpt2", dtype=DTYPE, attention_backend="paged").to(DEVICE).eval()
    runner = GPT2Runner(model, pool)
    engine = Engine(EngineConfig(num_blocks=256, block_size=16, max_num_seqs=8, max_num_batched_tokens=2048), runner)
    for r in reqs:
        engine.add_request(
            GenerateInput(request_id=r.rid, prompt="", max_new_tokens=r.max_new,
                          temperature=1.0, top_p=1.0, seed=0),
            prompt_token_ids=r.prompt_ids,
        )
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    engine.run_until_done()
    torch.cuda.synchronize()
    our_time = time.perf_counter() - t0

    hf_tps = total_out / hf_time
    our_tps = total_out / our_time
    speedup = hf_time / our_time

    print(f"\n  {'':>25} {'time (s)':>10} {'tok/s':>8} {'speedup':>9}")
    print(f"  {'-'*55}")
    print(f"  {'HF eager (sequential)':>25} {hf_time:>10.3f} {hf_tps:>8.0f} {'1.00x':>9}")
    print(f"  {'our engine (batched)':>25} {our_time:>10.3f} {our_tps:>8.0f} {speedup:>8.2f}x")

    return {
        "hf_time": hf_time, "our_time": our_time,
        "hf_tps": hf_tps, "our_tps": our_tps,
        "speedup": speedup, "total_out": total_out,
    }


def plot_throughput(data: Dict) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    bars = ax.bar(
        ["HF eager\n(sequential)", "Our engine\n(batched CB)"],
        [data["hf_tps"], data["our_tps"]],
        color=["#d62728", "#1f77b4"],
        alpha=0.85,
    )
    ax.set_ylabel("Output tokens / sec")
    ax.set_title(f"Aggregate throughput: 20 mixed-length requests\n(RTX 3060, fp16, GPT-2 124M)")
    for bar, val in zip(bars, [data["hf_tps"], data["our_tps"]]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 8,
                f"{val:.0f}", ha="center", fontsize=12, fontweight="bold")
    ax.text(1, data["our_tps"] * 0.5, f"{data['speedup']:.1f}x",
            ha="center", fontsize=18, fontweight="bold", color="white")
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    path = RESULTS_DIR / "throughput_comparison.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  saved {path}")


# ======================================================================
# Main
# ======================================================================


def main() -> None:
    if not torch.cuda.is_available():
        print("CUDA not available.", file=sys.stderr)
        sys.exit(1)

    print(f"GPU: {torch.cuda.get_device_name(DEVICE)}")
    print(f"dtype: {DTYPE}")

    # Group 1
    attn_data = run_attention_bench()
    plot_attention(attn_data)

    # Group 2
    paged_data = run_paged_decode_bench()
    plot_paged_decode(paged_data)

    # Group 3
    tp_data = run_throughput_bench()
    plot_throughput(tp_data)

    # Save raw numbers as JSON for the report
    all_data = {
        "gpu": torch.cuda.get_device_name(DEVICE),
        "dtype": str(DTYPE),
        "attention": attn_data,
        "paged_decode": paged_data,
        "throughput": tp_data,
    }
    json_path = RESULTS_DIR / "benchmark_data.json"
    with open(json_path, "w") as f:
        json.dump(all_data, f, indent=2, default=str)
    print(f"\n  saved raw data to {json_path}")
    print("\ndone. charts in results/")


if __name__ == "__main__":
    main()
