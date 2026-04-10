/*
 * CUDA paged decode-attention kernel.
 *
 * Equivalent to the Triton kernel in src/kernels/triton/paged_decode.py:
 * single-query attention that reads K/V directly from a paged KV cache
 * via a block-table indirection, using online softmax to avoid
 * materializing the full attention matrix.
 *
 * Grid:  (num_heads,)  — one thread block per head.
 * Block: (BLOCK_SIZE,) — one thread per slot within a KV block.
 *        Threads cooperate via shared-memory reductions for the
 *        softmax max, sum, and weighted-value accumulation.
 *
 * This kernel processes one sequence at a time. The Python wrapper
 * calls it once per sequence; a batched variant would extend the grid
 * to (num_heads, batch_size), which is a straightforward generalization.
 *
 * Memory access pattern:
 *   For each (head, logical_block) pair the kernel:
 *     1. Loads the physical block id from block_table[logical_block]
 *     2. Loads K[physical, slot, head, :] and V[physical, slot, head, :]
 *     3. Computes score = dot(q, k) * scale
 *     4. Updates online softmax state (running max m, running sum l,
 *        running weighted output acc)
 *   After all blocks: normalizes acc / l and writes the output.
 *
 * Why CUDA instead of Triton for this exercise:
 *   - Explicit shared-memory reductions (warp shuffles, __syncthreads)
 *   - Direct control over thread/block geometry
 *   - Understanding of the GPU execution model at a lower level
 *   - The kernel is functionally identical to the Triton version;
 *     the difference is in the level of abstraction, not the math
 */

#include <torch/extension.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cmath>

// Thread-block reduction: max across a warp using shuffle intrinsics.
__device__ float warp_reduce_max(float val) {
    for (int offset = 16; offset > 0; offset >>= 1)
        val = fmaxf(val, __shfl_xor_sync(0xffffffff, val, offset));
    return val;
}

// Thread-block reduction: sum across a warp using shuffle intrinsics.
__device__ float warp_reduce_sum(float val) {
    for (int offset = 16; offset > 0; offset >>= 1)
        val += __shfl_xor_sync(0xffffffff, val, offset);
    return val;
}

/*
 * _paged_decode_attention_kernel
 *
 * One thread block per head. Thread `tid` handles slot `tid` within
 * each KV block. We loop over block-table entries sequentially;
 * within each iteration every thread loads one K element, computes a
 * partial dot product, and we reduce across threads to get the full
 * score for that slot. Then we update the online softmax state.
 *
 * Template on HEAD_DIM so the inner dot-product loop is unrolled.
 */
template <int HEAD_DIM>
__global__ void _paged_decode_attention_kernel(
    const half* __restrict__ Q,        // [num_heads, HEAD_DIM]
    const half* __restrict__ K_cache,  // [num_layers, num_blocks, block_size, num_heads, HEAD_DIM]
    const half* __restrict__ V_cache,  // same shape
    half* __restrict__ Out,            // [num_heads, HEAD_DIM]
    const int* __restrict__ block_table,  // [max_num_blocks]
    int context_len,
    int layer_idx,
    float sm_scale,
    int num_blocks_pool,  // total blocks in the pool
    int block_size,       // slots per block
    int num_heads
) {
    const int head = blockIdx.x;
    const int tid = threadIdx.x;  // 0..block_size-1

    // Strides for the 5-D KV cache [num_layers, num_blocks, block_size, num_heads, HEAD_DIM]
    const int stride_layer = num_blocks_pool * block_size * num_heads * HEAD_DIM;
    const int stride_block = block_size * num_heads * HEAD_DIM;
    const int stride_slot  = num_heads * HEAD_DIM;
    const int stride_head  = HEAD_DIM;

    // Load Q for this head into registers.
    float q_reg[HEAD_DIM];
    const half* q_ptr = Q + head * HEAD_DIM;
    for (int d = 0; d < HEAD_DIM; d++) {
        q_reg[d] = __half2float(q_ptr[d]);
    }

    // Online softmax state.
    float m_i = -INFINITY;
    float l_i = 0.0f;
    float acc[HEAD_DIM];
    for (int d = 0; d < HEAD_DIM; d++) acc[d] = 0.0f;

    int num_kv_blocks = (context_len + block_size - 1) / block_size;

    for (int block_idx = 0; block_idx < num_kv_blocks; block_idx++) {
        int physical_block = block_table[block_idx];
        int position = block_idx * block_size + tid;
        bool valid = (tid < block_size) && (position < context_len);

        // Pointer to this slot's K and V in the cache.
        const half* k_ptr = K_cache + layer_idx * stride_layer
                            + physical_block * stride_block
                            + tid * stride_slot
                            + head * stride_head;
        const half* v_ptr = V_cache + layer_idx * stride_layer
                            + physical_block * stride_block
                            + tid * stride_slot
                            + head * stride_head;

        // Compute dot(q, k) for this slot.
        float score = 0.0f;
        if (valid) {
            for (int d = 0; d < HEAD_DIM; d++) {
                score += q_reg[d] * __half2float(k_ptr[d]);
            }
            score *= sm_scale;
        } else {
            score = -INFINITY;
        }

        // Shared-memory reduction to find the max score across all
        // threads (slots) in this block. This gives us m_block.
        float m_block = warp_reduce_max(score);
        // For block_size > 32, we'd need a cross-warp reduction via
        // shared memory. For block_size=16, one warp suffices.

        // Online softmax update.
        float m_new = fmaxf(m_i, m_block);
        float alpha = expf(m_i - m_new);
        float p = valid ? expf(score - m_new) : 0.0f;

        // Reduce sum(p) across threads.
        float p_sum = warp_reduce_sum(p);

        // Each thread accumulates its contribution to the output.
        // acc[d] = acc[d] * alpha + p * v[d] for this thread's slot.
        for (int d = 0; d < HEAD_DIM; d++) {
            acc[d] = acc[d] * alpha;
            if (valid) {
                acc[d] += p * __half2float(v_ptr[d]);
            }
        }

        l_i = l_i * alpha + p_sum;
        m_i = m_new;
    }

    // Reduce acc across threads: each thread holds one slot's
    // contribution; we need the sum across all slots.
    // For block_size=16 (one warp), use warp shuffles.
    for (int d = 0; d < HEAD_DIM; d++) {
        acc[d] = warp_reduce_sum(acc[d]);
    }

    // Thread 0 normalizes and writes the output.
    if (tid == 0) {
        half* out_ptr = Out + head * HEAD_DIM;
        for (int d = 0; d < HEAD_DIM; d++) {
            out_ptr[d] = __float2half(acc[d] / l_i);
        }
    }
}


// ---- C++ wrapper called from Python via torch.utils.cpp_extension ----

torch::Tensor paged_decode_attention_cuda(
    torch::Tensor q,            // [num_heads, HEAD_DIM], fp16
    torch::Tensor k_cache,      // [num_layers, num_blocks, block_size, num_heads, HEAD_DIM], fp16
    torch::Tensor v_cache,      // same
    torch::Tensor block_table,  // [max_num_blocks], int32
    int context_len,
    int layer_idx
) {
    TORCH_CHECK(q.is_cuda(), "q must be CUDA");
    TORCH_CHECK(q.scalar_type() == torch::kHalf, "q must be fp16");
    TORCH_CHECK(q.dim() == 2, "q must be [num_heads, head_dim]");

    int num_heads = q.size(0);
    int head_dim = q.size(1);
    int num_layers = k_cache.size(0);
    int num_blocks_pool = k_cache.size(1);
    int block_size = k_cache.size(2);
    float sm_scale = 1.0f / sqrtf((float)head_dim);

    auto out = torch::empty_like(q);

    dim3 grid(num_heads);
    dim3 block(block_size);

    // Dispatch on head_dim. GPT-2 uses 64.
    if (head_dim == 64) {
        _paged_decode_attention_kernel<64><<<grid, block>>>(
            (const half*)q.data_ptr<at::Half>(),
            (const half*)k_cache.data_ptr<at::Half>(),
            (const half*)v_cache.data_ptr<at::Half>(),
            (half*)out.data_ptr<at::Half>(),
            block_table.data_ptr<int>(),
            context_len, layer_idx, sm_scale,
            num_blocks_pool, block_size, num_heads
        );
    } else if (head_dim == 32) {
        _paged_decode_attention_kernel<32><<<grid, block>>>(
            (const half*)q.data_ptr<at::Half>(),
            (const half*)k_cache.data_ptr<at::Half>(),
            (const half*)v_cache.data_ptr<at::Half>(),
            (half*)out.data_ptr<at::Half>(),
            block_table.data_ptr<int>(),
            context_len, layer_idx, sm_scale,
            num_blocks_pool, block_size, num_heads
        );
    } else if (head_dim == 128) {
        _paged_decode_attention_kernel<128><<<grid, block>>>(
            (const half*)q.data_ptr<at::Half>(),
            (const half*)k_cache.data_ptr<at::Half>(),
            (const half*)v_cache.data_ptr<at::Half>(),
            (half*)out.data_ptr<at::Half>(),
            block_table.data_ptr<int>(),
            context_len, layer_idx, sm_scale,
            num_blocks_pool, block_size, num_heads
        );
    } else {
        TORCH_CHECK(false, "unsupported head_dim=", head_dim, "; expected 32, 64, or 128");
    }

    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("paged_decode_attention_cuda", &paged_decode_attention_cuda,
          "Paged decode attention (CUDA)");
}
