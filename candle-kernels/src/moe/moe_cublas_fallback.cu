/**
 *  @brief  cuBLAS-backed MoE GEMM fallback for pre-Ampere GPUs.
 *
 *  The WMMA MoE kernels (moe_wmma.cu) require compute capability >= 8.0
 *  because they instantiate bf16 WMMA fragments. On older GPUs (Volta,
 *  Turing, Pascal, ...) those kernels cannot even be loaded, so this
 *  file provides a cuBLAS-based alternative with the same logical result.
 *
 *  The strategy is:
 *
 *    1. Sort (done upstream: `sorted_token_ids` is already sorted by
 *       expert id, and `calculate_expert_offsets*` turns that into
 *       `[num_experts+1]` offsets).
 *
 *    2. **Gather**: copy each token's input row into a contiguous
 *       `A_packed [size_m, size_k]` buffer, in sorted-by-expert order.
 *       After this, expert `e`'s rows occupy
 *       `A_packed[offsets[e] .. offsets[e+1])`.
 *
 *    3. **Per-expert matmul** (driven from Rust): for each expert `e`,
 *       run one cuBLAS gemm `C_packed[e_slice] = A_packed[e_slice] @
 *       weights[e]^T`. This is done by the Rust layer because cuBLAS
 *       cannot be called from device code, and the per-expert M sizes
 *       need to be known on the host.
 *
 *    4. **Scatter**: for each row `i` in `[0, size_m)`, copy
 *       `C_packed[i]` back to `output[sorted_token_ids[i]]`, optionally
 *       multiplying by `topk_weights[sorted_token_ids[i]]`.
 *
 *  Both the gather and scatter kernels are plain memory movers — no
 *  Tensor Cores, no shared memory tricks. They compile on every CUDA
 *  arch we care about (Kepler and up).
 */

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cstdint>

#include "moe_utils.cuh"

namespace vllm_rs {

// ------------------------------------------------------------------
// dtype helpers
// ------------------------------------------------------------------

static __device__ __forceinline__ float to_float(const __half& h) {
    return __half2float(h);
}
static __device__ __forceinline__ float to_float(const __nv_bfloat16& b) {
    return __bfloat162float(b);
}

// ------------------------------------------------------------------
// gather: A_packed[i, :] = input[sorted_token_ids[i] / topk_divisor, :]
// ------------------------------------------------------------------
//
// One block per packed row. Threads within the block cooperatively
// stream the row using 128-bit (float4) vectorized loads where
// possible. `size_k` is asserted to be a multiple of 8 in Rust, so
// f16/bf16 vectorized copy with float4 is always aligned.

template <typename T>
__global__ void moe_gather_kernel(
    const T* __restrict__ input,
    const int32_t* __restrict__ sorted_token_ids,
    T* __restrict__ A_packed,
    int32_t size_m,
    int32_t size_k,
    int32_t topk_divisor
) {
    const int row = blockIdx.x;
    if (row >= size_m) return;

    const int32_t token_id = sorted_token_ids[row];
    const int32_t input_row = topk_divisor > 1 ? (token_id / topk_divisor) : token_id;

    const T* src = input + (size_t)input_row * (size_t)size_k;
    T* dst = A_packed + (size_t)row * (size_t)size_k;

    // 16 bytes / sizeof(T) = 8 for fp16/bf16
    constexpr int VEC_ELEMS = 16 / sizeof(T);
    const int vec_count = size_k / VEC_ELEMS;

    const float4* src_v = reinterpret_cast<const float4*>(src);
    float4* dst_v = reinterpret_cast<float4*>(dst);

    for (int i = threadIdx.x; i < vec_count; i += blockDim.x) {
        dst_v[i] = src_v[i];
    }

    // tail — size_k is asserted multiple of 8 so this is effectively dead
    // for f16/bf16, but keep it for safety if the assertion ever relaxes.
    const int tail = vec_count * VEC_ELEMS;
    for (int k = tail + threadIdx.x; k < size_k; k += blockDim.x) {
        dst[k] = src[k];
    }
}

// ------------------------------------------------------------------
// scatter: output[sorted_token_ids[i], :] = C_packed[i, :] * scale
// ------------------------------------------------------------------
//
// When `topk_weights` is nullptr the scale is 1 and we can use the
// same vectorized float4 path as gather. Otherwise we go element-wise
// through fp32 so the scale multiply is exact (matching the WMMA
// kernel which also accumulates in fp32 and calls `from_float` on
// `val * topk_weights[...]`).

template <typename T>
__global__ void moe_scatter_kernel(
    const T* __restrict__ C_packed,
    const int32_t* __restrict__ sorted_token_ids,
    const float* __restrict__ topk_weights,  // nullable
    T* __restrict__ output,
    int32_t size_m,
    int32_t size_n
) {
    const int row = blockIdx.x;
    if (row >= size_m) return;

    const int32_t token_id = sorted_token_ids[row];
    const T* src = C_packed + (size_t)row * (size_t)size_n;
    T* dst = output + (size_t)token_id * (size_t)size_n;

    if (topk_weights == nullptr) {
        // Pure copy — vectorized.
        constexpr int VEC_ELEMS = 16 / sizeof(T);
        const int vec_count = size_n / VEC_ELEMS;
        const float4* src_v = reinterpret_cast<const float4*>(src);
        float4* dst_v = reinterpret_cast<float4*>(dst);
        for (int i = threadIdx.x; i < vec_count; i += blockDim.x) {
            dst_v[i] = src_v[i];
        }
        const int tail = vec_count * VEC_ELEMS;
        for (int n = tail + threadIdx.x; n < size_n; n += blockDim.x) {
            dst[n] = src[n];
        }
    } else {
        const float scale = topk_weights[token_id];
        for (int n = threadIdx.x; n < size_n; n += blockDim.x) {
            float v = to_float(src[n]) * scale;
            from_float(dst[n], v);
        }
    }
}

}  // namespace vllm_rs

// ------------------------------------------------------------------
// FFI entry points
// ------------------------------------------------------------------

extern "C" void moe_cublas_gather(
    const void* input,                 // [rows, size_k]
    const int32_t* sorted_token_ids,   // [size_m]
    const int32_t* expert_ids,         // [size_m]
    void* a_packed,                    // [size_m, size_k] out
    int32_t* expert_counts,            // pre-alloc [num_experts]
    int32_t* expert_offsets,           // pre-alloc [num_experts + 1]
    int num_experts,
    int topk_divisor,
    int size_m,
    int size_k,
    int data_type,                     // 0 = fp16, 1 = bf16
    bool is_prefill,
    int64_t stream_raw
) {
    cudaStream_t stream = (cudaStream_t)stream_raw;

    // 1. expert_offsets (same helpers the WMMA path uses).
    if (is_prefill) {
        calculate_expert_offsets(
            expert_ids, size_m, expert_counts, expert_offsets, num_experts, stream
        );
    } else {
        calculate_expert_offsets_light(
            expert_ids, size_m, expert_counts, expert_offsets, num_experts, stream
        );
    }

    // 2. gather — one block per row, 128 threads per block.
    dim3 grid((unsigned)size_m, 1, 1);
    dim3 block(128, 1, 1);

    if (data_type == 0) {
        vllm_rs::moe_gather_kernel<__half><<<grid, block, 0, stream>>>(
            reinterpret_cast<const __half*>(input),
            sorted_token_ids,
            reinterpret_cast<__half*>(a_packed),
            size_m, size_k, topk_divisor
        );
    } else {
        vllm_rs::moe_gather_kernel<__nv_bfloat16><<<grid, block, 0, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(input),
            sorted_token_ids,
            reinterpret_cast<__nv_bfloat16*>(a_packed),
            size_m, size_k, topk_divisor
        );
    }
}

extern "C" void moe_cublas_scatter(
    const void* c_packed,              // [size_m, size_n]
    const int32_t* sorted_token_ids,   // [size_m]
    const float* topk_weights,         // [size_m] or nullptr
    void* output,                      // [size_m, size_n]
    int size_m,
    int size_n,
    int data_type,
    int64_t stream_raw
) {
    cudaStream_t stream = (cudaStream_t)stream_raw;

    dim3 grid((unsigned)size_m, 1, 1);
    dim3 block(128, 1, 1);

    if (data_type == 0) {
        vllm_rs::moe_scatter_kernel<__half><<<grid, block, 0, stream>>>(
            reinterpret_cast<const __half*>(c_packed),
            sorted_token_ids,
            topk_weights,
            reinterpret_cast<__half*>(output),
            size_m, size_n
        );
    } else {
        vllm_rs::moe_scatter_kernel<__nv_bfloat16><<<grid, block, 0, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(c_packed),
            sorted_token_ids,
            topk_weights,
            reinterpret_cast<__nv_bfloat16*>(output),
            size_m, size_n
        );
    }
}
