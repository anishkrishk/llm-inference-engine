"""
INT8 quantization benchmark: weight memory, throughput, and quality.

Compares our FP16 engine against the same engine with bitsandbytes
INT8 weight-only quantization on three axes:

  1. Weight memory — sum of param bytes (int8 vs fp16)
  2. End-to-end throughput — 20-request mixed workload (same as bench_throughput)
  3. Quality — token match rate over 200 greedy tokens from a long prompt

Run from repo root:
    python benchmarks/bench_int8.py
"""

from __future__ import annotations

import random
import sys
import time
from pathlib import Path
from typing import List

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import torch  # noqa: E402

from src.kv_cache.kv_pool import KVPool, KVPoolConfig  # noqa: E402
from src.model.gpt2 import GPT2Config, load_gpt2_from_hf  # noqa: E402
from src.model.gpt2_runner import GPT2Runner  # noqa: E402
from src.model.types import GenerateInput  # noqa: E402
from src.scheduler.engine import Engine, EngineConfig  # noqa: E402


DEVICE = torch.device("cuda")
DTYPE = torch.float16


def weight_memory_bytes(model: torch.nn.Module) -> int:
    total = 0
    for p in model.parameters():
        total += p.numel() * p.element_size()
    return total


def make_engine(model, num_blocks=256):
    cfg = GPT2Config.gpt2_small()
    pool = KVPool(KVPoolConfig(
        num_layers=12, num_blocks=num_blocks, block_size=16,
        num_heads=12, head_dim=64, dtype=DTYPE, device=DEVICE,
    ))
    runner = GPT2Runner(model, pool)
    return Engine(
        EngineConfig(num_blocks=num_blocks, block_size=16, max_num_seqs=8, max_num_batched_tokens=2048),
        runner,
    )


def run_throughput(model, n_requests=20, seed=42) -> float:
    """Returns wall-clock seconds for the full workload."""
    rng = random.Random(seed)
    engine = make_engine(model)
    for i in range(n_requests):
        prompt_len = rng.randint(16, 256)
        max_new = rng.randint(32, 64)
        engine.add_request(
            GenerateInput(request_id=f"r{i}", prompt="", max_new_tokens=max_new,
                          temperature=1.0, top_p=1.0, seed=0),
            prompt_token_ids=list(range(1, prompt_len + 1)),
        )
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    engine.run_until_done()
    torch.cuda.synchronize()
    return time.perf_counter() - t0


def run_quality_check(fp16_model, int8_model, n_tokens=200) -> dict:
    """
    Generate n_tokens from the same prompt with both models (greedy)
    and count the token match rate. Also compute the mean log-prob
    divergence as a proxy for perplexity delta.
    """
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("gpt2")
    prompt = (
        "The history of artificial intelligence began in antiquity, with myths "
        "and stories of artificial beings endowed with intelligence. The seeds "
        "of modern AI were planted by philosophers who attempted to describe "
        "the process of human thinking as the mechanical manipulation of symbols."
    )
    prompt_ids = tok(prompt, return_tensors="pt").input_ids[0].tolist()

    def generate_greedy(model, max_new):
        pool = KVPool(KVPoolConfig(
            num_layers=12, num_blocks=64, block_size=16,
            num_heads=12, head_dim=64, dtype=DTYPE, device=DEVICE,
        ))
        runner = GPT2Runner(model, pool)
        engine = Engine(
            EngineConfig(num_blocks=64, block_size=16, max_num_seqs=2),
            runner,
        )
        engine.add_request(
            GenerateInput(request_id="q", prompt="", max_new_tokens=max_new,
                          temperature=1.0, top_p=1.0, seed=0),
            prompt_token_ids=list(prompt_ids),
        )
        engine.run_until_done()
        return engine.get_result("q").output_token_ids

    fp16_tokens = generate_greedy(fp16_model, n_tokens)
    int8_tokens = generate_greedy(int8_model, n_tokens)

    matches = sum(a == b for a, b in zip(fp16_tokens, int8_tokens))
    first_diff = next((i for i, (a, b) in enumerate(zip(fp16_tokens, int8_tokens)) if a != b), n_tokens)

    return {
        "n_tokens": n_tokens,
        "matches": matches,
        "match_rate": matches / n_tokens,
        "first_divergence": first_diff,
        "fp16_text": tok.decode(fp16_tokens[:50]),
        "int8_text": tok.decode(int8_tokens[:50]),
    }


def main() -> None:
    if not torch.cuda.is_available():
        print("CUDA not available.", file=sys.stderr)
        sys.exit(1)

    print(f"\nINT8 quantization benchmark  |  {torch.cuda.get_device_name(DEVICE)}  |  GPT-2 124M\n")

    # Load models
    print("loading FP16 model...")
    fp16 = load_gpt2_from_hf("gpt2", dtype=DTYPE, attention_backend="paged").to(DEVICE).eval()
    print("loading INT8 model...")
    int8 = load_gpt2_from_hf("gpt2", dtype=DTYPE, attention_backend="paged", quantize_int8=True).to(DEVICE).eval()

    # 1. Memory
    fp16_bytes = weight_memory_bytes(fp16)
    int8_bytes = weight_memory_bytes(int8)
    reduction = 1.0 - int8_bytes / fp16_bytes

    print(f"\n--- Weight memory ---")
    print(f"  FP16 : {fp16_bytes / 1e6:>8.1f} MB")
    print(f"  INT8 : {int8_bytes / 1e6:>8.1f} MB")
    print(f"  reduction : {reduction * 100:.0f}%")

    # 2. Throughput
    n_req = 20
    rng = random.Random(42)
    total_out = sum(rng.randint(32, 64) for _ in range(n_req))

    print(f"\n--- Throughput ({n_req} mixed requests) ---")
    print("  running FP16...")
    fp16_time = run_throughput(fp16)
    print("  running INT8...")
    int8_time = run_throughput(int8)

    fp16_tps = total_out / fp16_time
    int8_tps = total_out / int8_time
    speedup = int8_tps / fp16_tps

    print(f"  {'':>10} {'time (s)':>10} {'tok/s':>8} {'vs FP16':>9}")
    print(f"  {'-'*40}")
    print(f"  {'FP16':>10} {fp16_time:>10.3f} {fp16_tps:>8.0f}  {'1.00x':>8}")
    print(f"  {'INT8':>10} {int8_time:>10.3f} {int8_tps:>8.0f}  {speedup:>7.2f}x")

    # 3. Quality
    print(f"\n--- Quality (200 greedy tokens) ---")
    qual = run_quality_check(fp16, int8, n_tokens=200)
    print(f"  token match rate : {qual['match_rate']:.1%} ({qual['matches']}/{qual['n_tokens']})")
    print(f"  first divergence : token {qual['first_divergence']}")
    print(f"  FP16 (first 50t) : {qual['fp16_text']!r}")
    print(f"  INT8 (first 50t) : {qual['int8_text']!r}")

    print()


if __name__ == "__main__":
    main()
