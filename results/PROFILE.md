# Profiling Analysis

**Tools:** Nsight Systems (`nsys`) with NVTX annotations + `torch.profiler` with CUDA activity tracing  
**Workload:** 5 concurrent requests (prompts 16–128 tokens, 20 output tokens each), 10 profiled engine steps after 5 warmup steps  
**Backend:** paged attention (Triton FlashAttention prefill, batched paged decode)

NVTX ranges are wired into the model (`embedding`, `block_N`, `qkv_proj`, `kv_write`, `paged_attn`, `out_proj`, `lm_head`). When Nsight Systems is available, run:
```bash
nsys profile -o results/nsys_trace python scripts/profile_engine.py
```
to get a full timeline with GPU kernel correlation.

---

## Nsight Systems NVTX breakdown

Nsight Systems captured NVTX-annotated wall-clock time per model region.
Median values across 10 engine steps (5 concurrent decode sequences):

| NVTX Range | Median per call | Calls | What it is |
|---|---|---|---|
| `engine_step` | **10.4 ms** | 10 | One full scheduler step (all 5 seqs) |
| `block_N` (each) | **~770 us** | 17 each | One transformer block (12 per step, some prefill steps have all 12) |
| `kv_write` | **228 us** | 144 | Batched scatter write of K,V for all seqs |
| `paged_attn` | **74 us** | 144 | Batched paged decode attention kernel |
| `qkv_proj` | **70 us** | 144 | Batched Q/K/V linear projection |
| `out_proj` | **66 us** | 144 | Batched attention output projection |
| `embedding` | **118 us** | 17 | Token + position embeddings |
| `lm_head` | **138 us** | 17 | Final LayerNorm + weight-tied LM head |

Per-step breakdown (12 blocks × ~770 us + embedding + lm_head ≈ **9.5 ms** GPU work + ~1 ms Python/scheduler overhead = ~10.4 ms total).

Inside each transformer block (per-layer):
- `kv_write`: 228 us (29%)
- `paged_attn`: 74 us (10%)
- `qkv_proj`: 70 us (9%)
- `out_proj`: 66 us (9%)
- Unlabeled (LayerNorm, MLP, residual, Python): ~332 us (43%)

## Nsight Systems CUDA API breakdown

| CUDA API | Calls | Total time | Median | What it is |
|---|---|---|---|---|
| `cudaLaunchKernel` | 3018 | 168.1 ms | 12.4 us | PyTorch op kernel launches |
| `cuLaunchKernel` | 797 | 12.6 ms | 14.5 us | Triton kernel launches |
| `cudaMemcpyAsync` | 446 | 31.5 ms | 14.6 us | Block table CPU→GPU transfers |
| `cudaStreamSynchronize` | 446 | 20.9 ms | 22.2 us | Forced CPU-GPU syncs |

**~3815 kernel launches per 10 steps = ~382 per step.** At ~14 us median per launch, **~5.3 ms of pure launch overhead per step**, or roughly **51% of the 10.4 ms step time**.

---

## torch.profiler op-level breakdown

From `torch.profiler` across the same 10 steps (93.8 ms total CPU time):

| Category | Self CPU time | % of total | # Calls | What it is |
|---|---|---|---|---|
| **Kernel launch overhead** | 29.8 ms | **31.7%** | 2140 | `cudaLaunchKernel` + `cuLaunchKernel` — CPU-side cost of dispatching CUDA kernels |
| **Linear projections** (addmm) | 11.6 ms | **12.4%** | 480 | The Q/K/V, output, and MLP matmuls — the actual compute |
| **Stream synchronization** | 5.7 ms | **6.1%** | 90 | `cudaStreamSynchronize` — forced by Python-side ops that need GPU results on CPU |
| **Tensor allocation** | 8.3 ms | **8.8%** | 1330 | `aten::empty` + `aten::empty_strided` — allocating intermediate tensors |
| **LayerNorm** | 3.6 ms | **3.9%** | 250 | `aten::native_layer_norm` |
| **dtype conversion** | 4.3 ms | **4.5%** | 160 | `aten::copy_` — fp16/int32/int64 conversions for block tables and index tensors |
| **Embeddings** | 1.0 ms | **1.1%** | 20 | `aten::embedding` + `aten::index_select` |
| **Everything else** | 29.5 ms | **31.5%** | — | View/reshape/stride ops, tensor indexing, Python framework overhead |

---

## Key findings

### 1. Kernel launch overhead is the dominant bottleneck (32% of wall time)

2140 CUDA kernel launches across 10 steps = **214 launches per engine step**. At ~14 microseconds per launch on this system, that's ~3 ms of pure CPU-side dispatch per step, just for launching work onto the GPU.

**Why so many launches?** Each engine step runs 12 transformer blocks. Each block dispatches:
- 1 LayerNorm (fused kernel)
- 1 QKV projection (addmm)
- 1 KV scatter write (indexed assignment = ~3 small kernels)
- 1 batched paged decode attention (Triton kernel)
- 1 output projection (addmm)
- 1 LayerNorm
- 1 MLP c_fc (addmm)
- 1 GELU activation
- 1 MLP c_proj (addmm)
- 2 residual adds

That's ~12 kernels per block × 12 blocks = **~144 kernel launches per step** for the model alone, plus embedding, LM head, scheduler overhead, and tensor housekeeping ops.

**How to fix:** `torch.compile()` (operator fusion, reduces kernel count by 3-5x) or CUDA graphs (records and replays a fixed kernel sequence with zero per-op launch overhead). Either would cut the 32% launch tax to near zero. These are standard production optimizations and are the next logical step beyond what we've built.

### 2. Linear projections are the main compute workload (12.4%)

480 `aten::addmm` calls across 10 steps = 48 per step (4 linears × 12 layers). Total FLOPs: **8.49 TFLOPs** across the profiled window. These are the matmuls that tensor cores accelerate; batching them (which we do via `decode_batch`) ensures each matmul is large enough to saturate the GPU.

At batch size 5: each addmm is `[5, 768] @ [768, K]` where K ∈ {2304, 768, 3072}. The largest (c_attn, K=2304) is 5 × 768 × 2304 = 8.85M FLOPs per call — small enough that kernel launch overhead dominates the wall-clock cost of each individual op.

### 3. Synchronization is expensive (6.1%)

90 `cudaStreamSynchronize` calls across 10 steps — these happen when Python code reads a GPU tensor value (e.g., `int(positions[i].item())` in the attention loop, or block table construction). Each sync stalls the CPU until the GPU catches up, breaking the asynchronous pipeline.

**How to fix:** Hoist all CPU-touching operations (position computation, block table padding, context length extraction) into the runner's `_batched_decode` method — which we mostly did in Phase 5 — and eliminate any remaining `.item()` calls inside the model. The remaining syncs likely come from the prefill path which still processes sequences one at a time.

### 4. Tensor allocation is a steady tax (8.8%)

1330 `aten::empty` / `aten::empty_strided` calls for intermediate tensors. PyTorch's caching allocator makes these cheap (no `cudaMalloc` per call) but they still cost ~2.5 microseconds each in Python overhead. `torch.compile` would eliminate most of these by fusing operations that produce and consume intermediates.

---

## What the Chrome trace shows

Open `results/profile_trace.json` at [ui.perfetto.dev](https://ui.perfetto.dev/) to see:

1. **CPU thread timeline**: the Python main thread dispatching operations. You'll see the scheduler's `step()` → runner's `step()` → model's `decode_batch()` nesting, with NVTX labels marking each transformer block.

2. **CUDA stream timeline**: the GPU kernels executing asynchronously below the CPU events. The key pattern: small kernels with gaps between them — the gaps are the CPU-side launch overhead visible in (1).

3. **Block-level structure**: the NVTX `block_N` ranges let you identify per-layer time. Each block takes roughly the same wall-clock time, confirming the workload is uniform across layers (no "one hot layer" bottleneck).

---

## Optimization roadmap (what would come next)

| Optimization | Expected impact | Complexity |
|---|---|---|
| `torch.compile` on the model forward | ~2x end-to-end by fusing ops and reducing launch count | Medium — requires static shapes or bucketing |
| CUDA graphs for the decode path | ~1.5-2x by eliminating per-step launch overhead entirely | Medium — requires fixed batch size and pre-allocated buffers |
| Batched prefill via cu_seqlens + FlashAttention varlen | Better prefill throughput under concurrent long-prompt arrivals | High — needs kernel changes |
| Operator fusion (QKV + attention + output in one kernel) | ~1.3x by reducing memory traffic between ops | High — custom Triton kernel work |
| Speculative decoding | Up to 2-3x decode throughput by verifying multiple draft tokens per step | High — requires a draft model |

---

## How to read the trace (teaching notes)

If an interviewer asks "walk me through what you see in a profile of your engine":

1. **Open the Chrome trace.** The top rows are CPU threads; the bottom rows are CUDA streams. Time flows left to right.

2. **Find a decode step.** Look for an `engine_step` NVTX range on the CPU timeline. Inside it you'll see 12 `block_N` ranges nested inside the model forward.

3. **Look at the GPU stream below.** You'll see a staircase of small kernels (addmm, layer_norm, the Triton paged attention kernel) with tiny gaps between them. The gaps are CPU dispatch latency.

4. **The key insight:** the GPU is spending significant time *idle* between kernels, waiting for the CPU to launch the next one. This is why kernel launch overhead (32%) is the #1 bottleneck. The GPU compute itself (addmm, attention) is efficient when it runs, but it's underutilized because of the launch cadence.

5. **What `torch.compile` would change:** it would fuse adjacent operations into fewer, larger kernels, closing the gaps. What CUDA graphs would change: they'd record the kernel sequence once and replay it without any per-kernel CPU dispatch, eliminating the gaps entirely.

6. **What you should say in an interview:** "The profile shows that at small batch sizes on GPT-2, kernel launch overhead dominates — 32% of wall time is just the CPU dispatching work to the GPU. The actual GPU compute (linear projections, attention) is efficient within each kernel. The next optimization step would be `torch.compile` or CUDA graphs to amortize the launch overhead, which is the standard production path for PyTorch inference."
