// Flash Attention decode kernel for GQA (Grouped Query Attention).
//
// Flash Decoding: single-token query attending to a KV cache.
// Supports GQA where n_kv_groups Q heads share one KV head.
//
// Tensor layout (candle convention, transposed from standard):
//   Q:   [1, n_q_heads, 1, head_dim]   - BF16, stride [n_q*head_dim, head_dim, head_dim, 1]
//   K:   [1, n_kv_heads, kv_len, head_dim] - BF16
//   V:   [1, n_kv_heads, kv_len, head_dim] - BF16
//   Out: [1, n_kv_heads*n_kv_groups, head_dim] - F32 (GQA output, reshaped by caller)
//
// Grid:  (n_kv_heads * n_kv_groups, 1, 1)  — one block per Q head
// Block: (D, 1, 1) — one thread per output dimension

#include "cuda_bf16.h"
#include <stdint.h>
#include <float.h>

#define FA_TILE 32

__device__ __forceinline__ float bf16_to_f32(__nv_bfloat16 x) {
    return __bfloat162float(x);
}

__device__ __forceinline__ float warp_reduce_sum(float val) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        val += __shfl_xor_sync(0xffffffff, val, offset);
    return val;
}

// Main Flash Attention decode kernel, templated on head_dim D.
//
// Each block handles exactly one Q head:
//   kv_head = blockIdx.x / n_kv_groups
//   q_group = blockIdx.x % n_kv_groups
//
// All threads in the block collaborate via shared memory.
template <int D>
static __device__ void flash_attn_decode_bf16_impl(
    const __nv_bfloat16* __restrict__ Q,
    const __nv_bfloat16* __restrict__ K,
    const __nv_bfloat16* __restrict__ V,
    float* __restrict__ out,
    int n_kv_groups,
    int kv_len,
    float scale
) {
    // Each block handles one Q head.
    const int q_head  = blockIdx.x;  // 0 .. n_q_heads-1
    const int kv_head = q_head / n_kv_groups;
    const int d       = threadIdx.x; // 0 .. D-1

    // Q pointer for this head: Q[q_head, 0, d]
    const __nv_bfloat16* q_ptr = Q + q_head * D;
    // K/V pointers for this KV head: K[kv_head, :, :]
    const __nv_bfloat16* k_ptr = K + kv_head * kv_len * D;
    const __nv_bfloat16* v_ptr = V + kv_head * kv_len * D;

    // Load Q value for this thread's dimension.
    float q_val = bf16_to_f32(q_ptr[d]);

    // Shared memory for KV tile and scores.
    __shared__ float smem_k[FA_TILE][D > 32 ? 1 : 32]; // not used for D>32 with warp reduce
    __shared__ float smem_scores[FA_TILE];
    __shared__ float smem_v[FA_TILE];  // V[j][d] for accumulation

    float acc   = 0.0f;
    float m_old = -FLT_MAX;
    float l_old = 0.0f;

    for (int tile_start = 0; tile_start < kv_len; tile_start += FA_TILE) {
        int tile_end  = tile_start + FA_TILE < kv_len ? tile_start + FA_TILE : kv_len;
        int tile_size = tile_end - tile_start;

        // Compute scores for this tile.
        // Each thread handles dimension d; we reduce across dimensions.
        for (int j = 0; j < tile_size; j++) {
            int pos = tile_start + j;
            float k_val = bf16_to_f32(k_ptr[pos * D + d]);
            float partial = q_val * k_val;

            // Sum across D dimensions.
            // For D threads in a block, use warp reduction (works for D <= 1024).
            float score = 0.0f;
            if (D <= 32) {
                score = warp_reduce_sum(partial);
            } else {
                // Multi-warp: each warp reduces its 32 elements, then warp 0 accumulates.
                float warp_sum = warp_reduce_sum(partial);
                if ((d & 31) == 0) {
                    // Lane 0 of each warp: atomicAdd to shared memory scratch.
                    // Use smem_scores[j] as scratch (zero first time).
                }
                // This approach requires separate scratch; use a simpler tactic:
                // Store warp sums in smem and reduce serially by thread 0.
                __shared__ float warp_sums[8]; // up to D=256=8 warps
                if ((d & 31) == 0) warp_sums[d / 32] = warp_sum;
                __syncthreads();
                if (d == 0) {
                    score = 0.0f;
                    int n_warps = (D + 31) / 32;
                    for (int w = 0; w < n_warps; w++) score += warp_sums[w];
                    smem_scores[j] = score * scale;
                }
                __syncthreads();
                score = smem_scores[j];
            }

            if (D <= 32 && d == 0) smem_scores[j] = score * scale;
        }
        if (D <= 32) __syncthreads();

        // Online softmax update.
        float m_new = m_old;
        for (int j = 0; j < tile_size; j++) m_new = fmaxf(m_new, smem_scores[j]);

        float exp_diff = __expf(m_old - m_new);
        float l_new = l_old * exp_diff;
        acc = acc * exp_diff;

        for (int j = 0; j < tile_size; j++) {
            float e = __expf(smem_scores[j] - m_new);
            l_new += e;
            acc += e * bf16_to_f32(v_ptr[(tile_start + j) * D + d]);
        }

        m_old = m_new;
        l_old = l_new;
        __syncthreads();
    }

    // Write output: out[q_head * D + d] = acc / l_old
    out[q_head * D + d] = (l_old > 0.0f) ? acc / l_old : 0.0f;
}

// The above shared-memory approach has issues for large D due to shared memory size.
// Use a cleaner implementation with proper shared memory layout.

template <int D>
static __device__ void flash_attn_decode_clean(
    const __nv_bfloat16* __restrict__ Q,
    const __nv_bfloat16* __restrict__ K,
    const __nv_bfloat16* __restrict__ V,
    float* __restrict__ out,
    int n_kv_groups,
    int kv_len,
    float scale
) {
    const int q_head  = blockIdx.x;
    const int kv_head = q_head / n_kv_groups;
    const int d       = threadIdx.x;

    const __nv_bfloat16* q_ptr = Q + q_head * D;
    const __nv_bfloat16* k_ptr = K + kv_head * kv_len * D;
    const __nv_bfloat16* v_ptr = V + kv_head * kv_len * D;

    float q_val = bf16_to_f32(q_ptr[d]);

    // Shared memory for warp sums and scores.
    __shared__ float smem_scores[FA_TILE];  // scores[j]
    extern __shared__ float smem_warp[];    // warp sums: (D/32) floats

    float acc   = 0.0f;
    float m_old = -FLT_MAX;
    float l_old = 0.0f;

    const int n_warps = (D + 31) / 32;

    for (int tile_start = 0; tile_start < kv_len; tile_start += FA_TILE) {
        int tile_end  = min(tile_start + FA_TILE, kv_len);
        int tile_size = tile_end - tile_start;

        // Compute scores[j] = (Q · K[j]) * scale  for j in tile.
        for (int j = 0; j < tile_size; j++) {
            int pos = tile_start + j;
            float k_val = bf16_to_f32(k_ptr[pos * D + d]);
            float partial = q_val * k_val;

            // Step 1: warp reduce within each warp.
            float warp_sum = warp_reduce_sum(partial);
            // Step 2: warp lane 0 writes to shared memory.
            if ((d & 31) == 0) smem_warp[d / 32] = warp_sum;
            __syncthreads();
            // Step 3: thread 0 reduces warp sums.
            if (d == 0) {
                float total = 0.0f;
                for (int w = 0; w < n_warps; w++) total += smem_warp[w];
                smem_scores[j] = total * scale;
            }
            __syncthreads();
        }

        // Online softmax update.
        float m_new = m_old;
        for (int j = 0; j < tile_size; j++) m_new = fmaxf(m_new, smem_scores[j]);

        float exp_diff = __expf(m_old - m_new);
        float l_new = l_old * exp_diff;
        acc = acc * exp_diff;

        for (int j = 0; j < tile_size; j++) {
            float e = __expf(smem_scores[j] - m_new);
            l_new += e;
            acc += e * bf16_to_f32(v_ptr[(tile_start + j) * D + d]);
        }

        m_old = m_new;
        l_old = l_new;
        __syncthreads();
    }

    out[q_head * D + d] = (l_old > 0.0f) ? acc / l_old : 0.0f;
}

// Kernel wrappers: one per supported head dimension.
// Each kernel is a separate global function (no template in extern "C").
// Dynamic shared memory size = (D/32) * sizeof(float) for warp sums.

#define DEF_FA_KERNEL(D_VAL)                                                    \
extern "C" __global__                                                           \
__launch_bounds__(D_VAL)                                                        \
void flash_attn_decode_bf16_d##D_VAL(                                           \
    const __nv_bfloat16* Q,                                                     \
    const __nv_bfloat16* K,                                                     \
    const __nv_bfloat16* V,                                                     \
    float* out,                                                                 \
    int n_kv_groups,                                                            \
    int kv_len,                                                                 \
    float scale                                                                 \
) {                                                                             \
    flash_attn_decode_clean<D_VAL>(Q, K, V, out, n_kv_groups, kv_len, scale);  \
}

DEF_FA_KERNEL(64)
DEF_FA_KERNEL(128)
DEF_FA_KERNEL(256)
DEF_FA_KERNEL(512)
