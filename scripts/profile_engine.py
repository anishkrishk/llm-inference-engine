"""
Profile the inference engine and produce detailed GPU kernel traces.

Captures a torch.profiler trace during a short generation workload
(5 concurrent requests, 20 tokens each) with NVTX annotations on the
model forward path. Produces:

  results/profile_trace.json   — Chrome-compatible trace viewable at
                                 chrome://tracing or ui.perfetto.dev
  results/profile_summary.txt  — top-level kernel time breakdown
  results/PROFILE.md           — written interpretation of the trace

The NVTX ranges (embedding, block_N, qkv_proj, kv_write, paged_attn,
out_proj, lm_head) label the key sections. When nsys is available,
run:
    nsys profile -o results/nsys_trace python scripts/profile_engine.py
to capture an nsys report with the same NVTX annotations.

Run from repo root:
    python scripts/profile_engine.py
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import torch  # noqa: E402
from torch.profiler import ProfilerActivity, profile, schedule  # noqa: E402

from src.kv_cache.kv_pool import KVPool, KVPoolConfig  # noqa: E402
from src.model.gpt2 import GPT2Config, load_gpt2_from_hf  # noqa: E402
from src.model.gpt2_runner import GPT2Runner  # noqa: E402
from src.model.types import GenerateInput  # noqa: E402
from src.scheduler.engine import Engine, EngineConfig  # noqa: E402


RESULTS_DIR = _REPO_ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)


def main() -> None:
    if not torch.cuda.is_available():
        print("CUDA not available.", file=sys.stderr)
        sys.exit(1)

    device = torch.device("cuda")
    dtype = torch.float16

    print("loading model...")
    gpt2_cfg = GPT2Config.gpt2_small()
    pool_cfg = KVPoolConfig(
        num_layers=12, num_blocks=128, block_size=16,
        num_heads=12, head_dim=64, dtype=dtype, device=device,
    )
    kv_pool = KVPool(pool_cfg)
    model = load_gpt2_from_hf("gpt2", dtype=dtype, attention_backend="paged")
    model = model.to(device).eval()
    runner = GPT2Runner(model, kv_pool)
    engine = Engine(
        EngineConfig(num_blocks=128, block_size=16, max_num_seqs=8),
        runner,
    )

    # Submit 5 requests with varied prompt lengths.
    prompts = {
        "r0": (list(range(1, 33)), 20),     # prompt_len=32
        "r1": (list(range(1, 65)), 20),     # prompt_len=64
        "r2": (list(range(1, 129)), 20),    # prompt_len=128
        "r3": (list(range(1, 17)), 20),     # prompt_len=16
        "r4": (list(range(1, 49)), 20),     # prompt_len=48
    }
    for rid, (pids, max_new) in prompts.items():
        engine.add_request(
            GenerateInput(
                request_id=rid, prompt="", max_new_tokens=max_new,
                temperature=1.0, top_p=1.0, seed=0,
            ),
            prompt_token_ids=pids,
        )

    # Warmup: run a few steps outside the profiler.
    print("warming up (5 steps)...")
    for _ in range(5):
        if engine.has_work():
            engine.step()

    # Profile the next N steps.
    n_profile_steps = 10
    print(f"profiling {n_profile_steps} steps...")

    trace_path = str(RESULTS_DIR / "profile_trace")

    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
        with_stack=False,
        with_flops=True,
    ) as prof:
        for _ in range(n_profile_steps):
            if engine.has_work():
                torch.cuda.nvtx.range_push("engine_step")
                engine.step()
                torch.cuda.nvtx.range_pop()

    # Export Chrome trace.
    chrome_path = RESULTS_DIR / "profile_trace.json"
    prof.export_chrome_trace(str(chrome_path))
    print(f"saved Chrome trace to {chrome_path}")
    print("  -> open at chrome://tracing or ui.perfetto.dev\n")

    # Print summary tables.
    summary_path = RESULTS_DIR / "profile_summary.txt"

    # Top CUDA kernels by total time.
    cuda_summary = prof.key_averages().table(
        sort_by="cuda_time_total", row_limit=25,
    )
    # Top ops by self CUDA time.
    self_summary = prof.key_averages().table(
        sort_by="self_cuda_time_total", row_limit=25,
    )

    with open(summary_path, "w") as f:
        f.write(f"Profile summary — {n_profile_steps} engine steps\n")
        f.write(f"{'=' * 60}\n\n")
        f.write("Top 25 ops by total CUDA time:\n")
        f.write(cuda_summary)
        f.write("\n\nTop 25 ops by self CUDA time:\n")
        f.write(self_summary)

    print(f"saved summary to {summary_path}\n")
    print("Top 15 by self CUDA time:")
    print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=15))


if __name__ == "__main__":
    main()
