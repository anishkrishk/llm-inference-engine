# llm-inference-engine

A single-GPU LLM inference engine built from scratch: paged KV cache,
continuous batching, custom Triton and CUDA attention kernels, and a
streaming gRPC service. Targets GPT-2 / TinyLlama on consumer hardware
(RTX 3060, 12 GB).

## Quick start

```bash
# 1. Create a virtualenv and install dependencies (under WSL2 with CUDA).
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 2. Generate gRPC stubs.
make gen-proto

# 3. Run the test suite.
make test
```

## Layout

```
src/
  kv_cache/    Paged KV cache: block allocator, per-sequence block tables
  scheduler/   Iteration-level continuous batcher
  model/       Model runner, weight loading, sampling
  kernels/     Triton (prefill) and CUDA (decode) attention kernels
  service/     gRPC server and streaming
benchmarks/    Throughput / latency / kernel microbenchmarks
results/       Generated plots and reports
```
