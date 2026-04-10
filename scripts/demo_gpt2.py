"""
End-to-end GPT-2 generation demo driving our engine.

Loads the 124M GPT-2 checkpoint from HuggingFace, instantiates a paged
KV pool, wires them into our scheduler, and greedy-decodes a short
completion. Reports a realistic throughput number measured on the
actual hardware.

Run from repo root:
    python scripts/demo_gpt2.py "your prompt here"
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import torch  # noqa: E402

from src.kv_cache.kv_pool import KVPool, KVPoolConfig  # noqa: E402
from src.model.gpt2 import GPT2Config, load_gpt2_from_hf  # noqa: E402
from src.model.gpt2_runner import GPT2Runner  # noqa: E402
from src.model.types import GenerateInput  # noqa: E402
from src.scheduler.engine import Engine, EngineConfig  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "prompt", nargs="?", default="The quick brown fox",
        help="prompt text to complete",
    )
    parser.add_argument("--max-new", type=int, default=40)
    parser.add_argument("--num-blocks", type=int, default=64)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument(
        "--backend", choices=["eager", "triton"], default="eager",
        help="attention kernel backend",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("CUDA not available; this demo requires a GPU.", file=sys.stderr)
        sys.exit(1)

    device = torch.device("cuda")
    dtype = torch.float16

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    prompt_ids = tokenizer(args.prompt, return_tensors="pt").input_ids[0].tolist()

    print(f"loading gpt2 (124M) in fp16 on {device} (attention backend: {args.backend})...")
    t0 = time.perf_counter()
    model = load_gpt2_from_hf(
        "gpt2", dtype=dtype, attention_backend=args.backend
    ).to(device).eval()
    load_s = time.perf_counter() - t0

    gpt2_cfg = GPT2Config.gpt2_small()
    pool_cfg = KVPoolConfig(
        num_layers=gpt2_cfg.num_layers,
        num_blocks=args.num_blocks,
        block_size=args.block_size,
        num_heads=gpt2_cfg.num_heads,
        head_dim=gpt2_cfg.head_dim,
        dtype=dtype,
        device=device,
    )
    kv_pool = KVPool(pool_cfg)
    runner = GPT2Runner(model, kv_pool)
    engine = Engine(
        EngineConfig(
            num_blocks=pool_cfg.num_blocks,
            block_size=pool_cfg.block_size,
            max_num_seqs=4,
            max_num_batched_tokens=1024,
        ),
        runner,
    )

    pool_mb = pool_cfg.total_bytes() / 1e6
    token_cap = pool_cfg.num_blocks * pool_cfg.block_size
    print(
        f"model loaded in {load_s:.2f}s; "
        f"kv pool: {pool_cfg.num_blocks} blocks x {pool_cfg.block_size} slots "
        f"({token_cap} token capacity, {pool_mb:.1f} MB)"
    )
    print()

    engine.add_request(
        GenerateInput(
            request_id="demo",
            prompt=args.prompt,
            max_new_tokens=args.max_new,
            temperature=1.0,
            top_p=1.0,
            seed=0,
        ),
        prompt_token_ids=prompt_ids,
    )

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    engine.run_until_done()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    result = engine.get_result("demo")
    output_tokens = result.output_token_ids
    output_text = tokenizer.decode(output_tokens)

    tok_per_s = len(output_tokens) / elapsed if elapsed > 0 else float("inf")
    print(f"prompt : {args.prompt}")
    print(f"output : {output_text}")
    print()
    print(
        f"generated {len(output_tokens)} tokens in {elapsed * 1000:.1f} ms "
        f"({tok_per_s:.0f} tok/s)"
    )


if __name__ == "__main__":
    main()
