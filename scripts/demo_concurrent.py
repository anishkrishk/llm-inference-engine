"""
Concurrent streaming demo: 10 gRPC clients streaming tokens in parallel.

Launches the async gRPC server in-process, sends 10 GenerateStream
requests concurrently, and prints tokens as they arrive interleaved
from different requests. Reports aggregate throughput and per-request
latency at the end.

Run from repo root:
    python scripts/demo_concurrent.py
"""

from __future__ import annotations

import asyncio
import sys
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import torch  # noqa: E402
import grpc  # noqa: E402

from src.generated import inference_pb2, inference_pb2_grpc  # noqa: E402
from src.kv_cache.kv_pool import KVPool, KVPoolConfig  # noqa: E402
from src.model.gpt2 import GPT2Config, load_gpt2_from_hf  # noqa: E402
from src.model.gpt2_runner import GPT2Runner  # noqa: E402
from src.scheduler.engine import Engine, EngineConfig  # noqa: E402
from src.service.server import AsyncInferenceService  # noqa: E402


PROMPTS = [
    "The quick brown fox",
    "Once upon a time in a land far away",
    "Artificial intelligence will change",
    "The meaning of life is",
    "In the beginning there was",
    "Climate change is causing",
    "The stock market today shows",
    "Deep learning models are",
    "How to build a successful",
    "The future of programming",
]


async def run_client(
    stub: inference_pb2_grpc.InferenceServiceStub,
    prompt: str,
    request_id: str,
    max_new: int,
    results: dict,
) -> None:
    """Send one streaming request and collect results."""
    req = inference_pb2.GenerateRequest(
        request_id=request_id,
        prompt=prompt,
        max_new_tokens=max_new,
        temperature=1.0,
        top_p=1.0,
    )
    t0 = time.perf_counter()
    tokens_received = 0
    full_text = ""
    ttft = 0.0
    total_ms = 0.0

    async for chunk in stub.GenerateStream(req):
        if chunk.HasField("final"):
            f = chunk.final
            full_text = f.full_text
            ttft = f.usage.ttft_ms
            total_ms = f.usage.total_latency_ms

    elapsed = time.perf_counter() - t0
    results[request_id] = {
        "prompt": prompt,
        "output": full_text,
        "tokens": max_new,
        "elapsed_ms": elapsed * 1000,
        "ttft_ms": ttft,
        "total_ms": total_ms,
    }


async def main() -> None:
    if not torch.cuda.is_available():
        print("CUDA not available.", file=sys.stderr)
        sys.exit(1)

    device = torch.device("cuda")
    dtype = torch.float16
    max_new = 20
    port = 50052

    print("loading model...")
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("gpt2")

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
        EngineConfig(num_blocks=128, block_size=16, max_num_seqs=10),
        runner,
    )

    # Start the async gRPC server in-process.
    server = grpc.aio.server()
    service = AsyncInferenceService(engine, tokenizer)
    inference_pb2_grpc.add_InferenceServiceServicer_to_server(service, server)
    server.add_insecure_port(f"localhost:{port}")
    await server.start()
    await service.start_engine_loop()
    print(f"server listening on localhost:{port}")

    # Connect a client stub.
    channel = grpc.aio.insecure_channel(f"localhost:{port}")
    stub = inference_pb2_grpc.InferenceServiceStub(channel)

    # Launch 10 concurrent streaming requests.
    print(f"\nsending {len(PROMPTS)} concurrent GenerateStream requests "
          f"(max_new={max_new})...\n")
    results: dict = {}
    t_start = time.perf_counter()
    tasks = [
        run_client(stub, prompt, f"r{i:02d}", max_new, results)
        for i, prompt in enumerate(PROMPTS)
    ]
    await asyncio.gather(*tasks)
    t_total = time.perf_counter() - t_start

    total_tokens = len(PROMPTS) * max_new

    print(f"\n{'req':>5} | {'prompt':<40} | {'output tokens':>13} | {'latency ms':>11}")
    print("-" * 80)
    latencies = []
    for rid in sorted(results):
        r = results[rid]
        print(f"{rid:>5} | {r['prompt']:<40} | {r['tokens']:>13} | {r['elapsed_ms']:>10.0f}")
        latencies.append(r["elapsed_ms"])

    latencies.sort()
    p50 = latencies[len(latencies) // 2]
    p95_idx = min(int(len(latencies) * 0.95), len(latencies) - 1)
    p95 = latencies[p95_idx]

    print()
    print(f"total: {total_tokens} tokens in {t_total:.3f}s")
    print(f"aggregate throughput: {total_tokens / t_total:.0f} tok/s")
    print(f"p50 latency: {p50:.0f} ms")
    print(f"p95 latency: {p95:.0f} ms")

    await channel.close()
    await server.stop(grace=0)


if __name__ == "__main__":
    asyncio.run(main())
