"""
Throughput comparison: our engine (batched continuous batching) vs
HuggingFace's model.generate() processing the same requests sequentially.

This benchmark produces the "Xx throughput" number for the resume.

Workload: 20 requests with prompt lengths 16–256 and max_new_tokens
32–64, drawn from a seeded RNG for reproducibility. The distribution
is intentionally mixed: continuous batching's advantage comes from
filling decode slots with new prefills as short requests finish,
avoiding the "wait for the longest request in the batch" problem
that plagues static batching.

Run from repo root:
    python benchmarks/bench_throughput.py
"""

from __future__ import annotations

import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import torch  # noqa: E402


@dataclass
class BenchRequest:
    request_id: str
    prompt_ids: List[int]
    max_new: int


def build_requests(n: int, seed: int = 42) -> List[BenchRequest]:
    rng = random.Random(seed)
    reqs: List[BenchRequest] = []
    for i in range(n):
        prompt_len = rng.randint(16, 256)
        max_new = rng.randint(32, 64)
        # Use sequential token ids as fake prompt; content doesn't matter
        # for throughput measurement.
        prompt_ids = list(range(1, prompt_len + 1))
        reqs.append(BenchRequest(f"r{i:02d}", prompt_ids, max_new))
    return reqs


def run_hf_baseline(reqs: List[BenchRequest], device, dtype) -> float:
    """Run HF model.generate() one request at a time. Returns wall-clock seconds."""
    from transformers import GPT2LMHeadModel

    hf = GPT2LMHeadModel.from_pretrained("gpt2").to(device=device, dtype=dtype).eval()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.inference_mode():
        for r in reqs:
            ids = torch.tensor([r.prompt_ids], device=device)
            hf.generate(ids, max_new_tokens=r.max_new, do_sample=False, pad_token_id=50256)
    torch.cuda.synchronize()
    return time.perf_counter() - t0


def run_our_engine(reqs: List[BenchRequest], device, dtype) -> float:
    """Run our engine with continuous batching. Returns wall-clock seconds."""
    from src.kv_cache.kv_pool import KVPool, KVPoolConfig
    from src.model.gpt2 import GPT2Config, load_gpt2_from_hf
    from src.model.gpt2_runner import GPT2Runner
    from src.model.types import GenerateInput
    from src.scheduler.engine import Engine, EngineConfig

    gpt2_cfg = GPT2Config.gpt2_small()
    pool_cfg = KVPoolConfig(
        num_layers=gpt2_cfg.num_layers,
        num_blocks=256,
        block_size=16,
        num_heads=gpt2_cfg.num_heads,
        head_dim=gpt2_cfg.head_dim,
        dtype=dtype,
        device=device,
    )
    kv_pool = KVPool(pool_cfg)
    model = load_gpt2_from_hf("gpt2", dtype=dtype, attention_backend="paged")
    model = model.to(device).eval()
    runner = GPT2Runner(model, kv_pool)
    engine = Engine(
        EngineConfig(
            num_blocks=pool_cfg.num_blocks,
            block_size=pool_cfg.block_size,
            max_num_seqs=8,
            max_num_batched_tokens=2048,
        ),
        runner,
    )

    for r in reqs:
        engine.add_request(
            GenerateInput(
                request_id=r.request_id,
                prompt="",
                max_new_tokens=r.max_new,
                temperature=1.0,
                top_p=1.0,
                seed=0,
            ),
            prompt_token_ids=r.prompt_ids,
        )

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    engine.run_until_done()
    torch.cuda.synchronize()
    return time.perf_counter() - t0


def main() -> None:
    if not torch.cuda.is_available():
        print("CUDA not available.", file=sys.stderr)
        sys.exit(1)

    device = torch.device("cuda")
    dtype = torch.float16
    n_requests = 20

    reqs = build_requests(n_requests)
    total_output_tokens = sum(r.max_new for r in reqs)
    total_prompt_tokens = sum(len(r.prompt_ids) for r in reqs)

    print(
        f"\nthroughput comparison  |  {n_requests} requests  |  "
        f"{torch.cuda.get_device_name(device)}  |  fp16  |  GPT-2 124M\n"
    )
    print(f"  total prompt tokens  : {total_prompt_tokens}")
    print(f"  total output tokens  : {total_output_tokens}")
    prompt_lens = [len(r.prompt_ids) for r in reqs]
    gen_lens = [r.max_new for r in reqs]
    print(
        f"  prompt len range     : {min(prompt_lens)}–{max(prompt_lens)} "
        f"(mean {sum(prompt_lens)/len(prompt_lens):.0f})"
    )
    print(
        f"  max_new range        : {min(gen_lens)}–{max(gen_lens)} "
        f"(mean {sum(gen_lens)/len(gen_lens):.0f})"
    )
    print()

    # -- HF baseline --
    print("running HF baseline (sequential model.generate())...")
    hf_time = run_hf_baseline(reqs, device, dtype)
    hf_tps = total_output_tokens / hf_time

    # -- Our engine --
    print("running our engine (continuous batching, paged KV, batched decode)...")
    our_time = run_our_engine(reqs, device, dtype)
    our_tps = total_output_tokens / our_time

    speedup = hf_time / our_time

    print()
    print(f"{'':>25} {'wall-clock (s)':>15} {'output tok/s':>13} {'speedup':>9}")
    print("-" * 65)
    print(f"{'HF eager (sequential)':>25} {hf_time:>15.3f} {hf_tps:>13.0f} {'1.00x':>9}")
    print(f"{'our engine (batched)':>25} {our_time:>15.3f} {our_tps:>13.0f} {speedup:>8.2f}x")
    print()


if __name__ == "__main__":
    main()
