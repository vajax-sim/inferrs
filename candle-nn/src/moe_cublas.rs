// cuBLAS-backed MoE GEMM fallback used on pre-Ampere GPUs where the
// WMMA kernels in `moe_wmma.cu` cannot run. See
// `candle-kernels/src/moe/moe_cublas_fallback.cu` for the gather /
// scatter kernels and the overall design rationale.
//
// This module only targets the dense (non-GGUF) MoE path. GGUF prefill
// on pre-Ampere remains gated off — quantized weights would additionally
// need a dequant-then-cuBLAS path which is out of scope here.

#![cfg(feature = "cuda")]

use candle::cuda_backend::cudarc;
use candle::cuda_backend::kernels::ffi;
use candle::{DType, Result, Tensor};

use cudarc::cublas::sys;
use cudarc::driver::DevicePtr;

/// Entry point matching `moe_gemm`. Called from `candle-nn/src/moe.rs`
/// when `has_wmma_support()` returns false.
pub(crate) fn moe_gemm_cublas(
    input: &Tensor,
    weights: &Tensor,
    topk_weights: &Option<Tensor>,
    sorted_token_ids: &Tensor,
    experts_ids: &Tensor,
    topk: usize,
    is_prefill: bool,
) -> Result<Tensor> {
    match input.dtype() {
        DType::F16 => run::<half::f16>(
            input,
            weights,
            topk_weights,
            sorted_token_ids,
            experts_ids,
            topk,
            is_prefill,
            0, // data_type code matching moe_wmma.cu: 0 = fp16
            sys::cudaDataType_t::CUDA_R_16F,
        ),
        DType::BF16 => run::<half::bf16>(
            input,
            weights,
            topk_weights,
            sorted_token_ids,
            experts_ids,
            topk,
            is_prefill,
            1, // 1 = bf16
            sys::cudaDataType_t::CUDA_R_16BF,
        ),
        _ => candle::bail!("moe_gemm cuBLAS fallback only accepts f16/bf16 inputs"),
    }
}

fn run<T>(
    input: &Tensor,
    weights: &Tensor,
    topk_weights: &Option<Tensor>,
    sorted_token_ids: &Tensor,
    experts_ids: &Tensor,
    topk: usize,
    is_prefill: bool,
    data_type_code: i32,
    cublas_dtype: sys::cudaDataType_t,
) -> Result<Tensor>
where
    T: candle::cuda_backend::CudaDType + cudarc::driver::DeviceRepr,
{
    use candle::op::BackpropOp;
    use core::ffi::c_void;

    // -------- Shape bookkeeping (mirrors moe_gemm) --------
    let (mut size_m, size_k1) = input.dims2()?;
    if topk_weights.is_none() {
        size_m *= topk;
    }
    let (num_experts, size_n, size_k) = weights.dims3()?;
    assert!(
        size_k == size_k1,
        "input {:?} and weight {:?} last dim mismatch!",
        size_k1,
        size_k
    );
    assert!(
        size_k % 8 == 0,
        "moe_gemm cuBLAS fallback requires size_k divisible by 8 \
         (matches the WMMA path assumption for vectorized loads)"
    );

    let dev = input.device().as_cuda_device()?;
    let stream = dev.cuda_stream();
    let stream_raw = stream.cu_stream() as i64;

    // -------- Lock tensor storages + bind CudaSlice refs at function scope.
    // Mirrors the existing `moe.rs` pattern: the MutexGuard returned by
    // `storage_and_layout()` is held in a function-scoped `*_s` binding, the
    // `CudaSlice<T>` reference is shadowed, and `.device_ptr()` is only ever
    // called *inline* inside a FFI call expression. That keeps the
    // `(CUdeviceptr, SyncOnDrop)` tuple alive as a statement-temporary until
    // the `;` that ends the FFI call, which is what cudarc's docs require
    // ("drop the SyncOnDrop *after* the read of the CUdeviceptr is
    // scheduled"). --------
    let (input_s, _) = input.storage_and_layout();
    let input = match &*input_s {
        candle::Storage::Cuda(c) => c.as_cuda_slice::<T>()?,
        _ => candle::bail!("input must be a cuda tensor"),
    };

    let (weights_s, _) = weights.storage_and_layout();
    let weights = match &*weights_s {
        candle::Storage::Cuda(c) => c.as_cuda_slice::<T>()?,
        _ => candle::bail!("weights must be a cuda tensor"),
    };

    let (sorted_s, _) = sorted_token_ids.storage_and_layout();
    let sorted_token_ids = match &*sorted_s {
        candle::Storage::Cuda(c) => c.as_cuda_slice::<u32>()?,
        _ => candle::bail!("sorted_token_ids must be a cuda tensor"),
    };

    let (experts_s, _) = experts_ids.storage_and_layout();
    let experts_ids = match &*experts_s {
        candle::Storage::Cuda(c) => c.as_cuda_slice::<u32>()?,
        _ => candle::bail!("experts_ids must be a cuda tensor"),
    };

    // topk_weights is optional. Hold the storage guard and CudaSlice ref at
    // function scope in parallel `Option`s so they outlive the scatter call.
    let topk_weights_holder = match topk_weights {
        Some(tw) => {
            let (s, _) = tw.storage_and_layout();
            Some(s)
        }
        None => None,
    };
    let topk_weights_slice: Option<&cudarc::driver::CudaSlice<f32>> =
        match &topk_weights_holder {
            Some(s) => match &**s {
                candle::Storage::Cuda(c) => Some(c.as_cuda_slice::<f32>()?),
                _ => candle::bail!("topk_weights must be a cuda tensor"),
            },
            None => None,
        };

    // Divisor the gather kernel uses to map sorted_token_ids[i] back
    // to a row in the input tensor. Matches
    //   `int input_index = token_index / (topk_weights? 1: topk);`
    // from moe_wmma.cu.
    let topk_divisor: i32 = if topk_weights.is_some() {
        1
    } else {
        topk as i32
    };

    // -------- Scratch allocations --------
    let a_packed = unsafe { dev.alloc::<T>(size_m * size_k) }?;
    let c_packed = unsafe { dev.alloc::<T>(size_m * size_n) }?;
    let expert_counts = unsafe { dev.alloc::<u32>(num_experts) }?;
    let expert_offsets = unsafe { dev.alloc::<u32>(num_experts + 1) }?;
    let output = unsafe { dev.alloc::<T>(size_m * size_n) }?;

    // -------- Step 1: compute expert offsets + gather rows --------
    unsafe {
        ffi::moe_cublas_gather(
            input.device_ptr(input.stream()).0 as *const c_void,
            sorted_token_ids.device_ptr(sorted_token_ids.stream()).0 as *const i32,
            experts_ids.device_ptr(experts_ids.stream()).0 as *const i32,
            a_packed.device_ptr(a_packed.stream()).0 as *mut c_void,
            expert_counts.device_ptr(expert_counts.stream()).0 as *mut i32,
            expert_offsets.device_ptr(expert_offsets.stream()).0 as *mut i32,
            num_experts as i32,
            topk_divisor,
            size_m as i32,
            size_k as i32,
            data_type_code,
            is_prefill,
            stream_raw,
        );
    }

    // -------- Step 2: read expert offsets back to the host --------
    // cuBLAS cannot be launched from device code and per-expert GEMMs need
    // the host-side `rows_e` sizes. This is a sync point — the stream stalls
    // while the small `[num_experts + 1]` array comes back — but that's
    // unavoidable for a grouped GEMM driven from the host. It only runs on
    // pre-Ampere so we don't care about the cost.
    //
    // `clone_dtoh` issues `cuMemcpyDtoHAsync` under the hood, so the host
    // `Vec<u32>` is not safe to read until we explicitly synchronize the
    // stream.
    let offsets_host: Vec<u32> = dev.clone_dtoh(&expert_offsets)?;
    stream.synchronize().map_err(candle::Error::wrap)?;

    // -------- Step 3: per-expert cuBLAS gemm --------
    // Math we want (row-major):
    //   C[rows_e, N] = A[rows_e, K] @ W[e]^T
    // where W[e] is stored row-major as [N, K].
    //
    // cuBLAS is column-major, so via the standard row-major / col-major
    // swap this becomes the column-major call:
    //   op_a = OP_T  (W as [N, K] col-major → transposed → [K, N])
    //   op_b = OP_N  (A_packed_e as [K, rows_e] col-major, contiguous)
    //   m    = N     (col-major rows of C^T)
    //   n    = rows_e
    //   k    = K
    //   lda  = K     (physical stride of W[e])
    //   ldb  = K     (physical stride of A_packed_e)
    //   ldc  = N     (physical stride of C_packed_e)
    //
    // Using fp32 compute for accuracy that matches the WMMA kernel's
    // float accumulator. `CUBLAS_GEMM_DEFAULT` (not `*_TENSOR_OP`)
    // because this path only runs on GPUs that don't have Tensor Cores
    // for these dtypes anyway.
    let cublas = dev.cublas_handle();
    let handle = *cublas.handle();
    let alpha: f32 = 1.0;
    let beta: f32 = 0.0;
    let compute_type = sys::cublasComputeType_t::CUBLAS_COMPUTE_32F;
    let algo = sys::cublasGemmAlgo_t::CUBLAS_GEMM_DEFAULT;

    let elem_size = std::mem::size_of::<T>();
    let weight_expert_stride_bytes = (size_n * size_k * elem_size) as u64;

    // Hoist the `(base_ptr, SyncOnDrop)` tuples out of the loop. The base
    // pointers don't change across experts — only the byte offsets we add
    // on top. Binding them as named `_g_*` locals keeps the guards alive
    // for every `gemm_ex` call in the loop body; they drop at the end of
    // this enclosing block (after the loop), which records the
    // read/write events onto the cuda stream for subsequent allocator
    // cleanup.
    let (a_base, _g_a) = a_packed.device_ptr(a_packed.stream());
    let (c_base, _g_c) = c_packed.device_ptr(c_packed.stream());
    let (w_base, _g_w) = weights.device_ptr(weights.stream());

    for e in 0..num_experts {
        let start = offsets_host[e] as usize;
        let end = offsets_host[e + 1] as usize;
        let rows_e = end - start;
        if rows_e == 0 {
            continue;
        }

        let a_offset_bytes = (start * size_k * elem_size) as u64;
        let c_offset_bytes = (start * size_n * elem_size) as u64;
        let w_offset_bytes = (e as u64) * weight_expert_stride_bytes;

        let a_ptr = (a_base + a_offset_bytes) as *const c_void;
        let c_ptr = (c_base + c_offset_bytes) as *mut c_void;
        let w_ptr = (w_base + w_offset_bytes) as *const c_void;

        unsafe {
            cudarc::cublas::result::gemm_ex(
                handle,
                sys::cublasOperation_t::CUBLAS_OP_T, // W: transposed
                sys::cublasOperation_t::CUBLAS_OP_N, // A: no-trans
                size_n as i32,                       // m (col-major)
                rows_e as i32,                       // n
                size_k as i32,                       // k
                &alpha as *const f32 as *const c_void,
                w_ptr,
                cublas_dtype,
                size_k as i32, // lda
                a_ptr,
                cublas_dtype,
                size_k as i32, // ldb
                &beta as *const f32 as *const c_void,
                c_ptr,
                cublas_dtype,
                size_n as i32, // ldc
                compute_type,
                algo,
            )
            .map_err(candle::Error::wrap)?;
        }
    }

    // -------- Step 4: scatter packed outputs back --------
    //
    // The topk_weights pointer is optional, so we can't compute it as a
    // single inline expression inside the FFI arg list the way we do for
    // the other tensors. Instead we bind the `(ptr, SyncOnDrop)` tuple in
    // a local `Option` right above the FFI call; the named binding lives
    // through the end of its enclosing block, keeping the guard alive past
    // the scatter launch.
    let topk_weights_ptr_holder = topk_weights_slice
        .as_ref()
        .map(|s| s.device_ptr(s.stream()));
    let topk_weights_ptr: *const f32 = topk_weights_ptr_holder
        .as_ref()
        .map(|(p, _)| *p as *const f32)
        .unwrap_or(std::ptr::null());

    unsafe {
        ffi::moe_cublas_scatter(
            c_packed.device_ptr(c_packed.stream()).0 as *const c_void,
            sorted_token_ids.device_ptr(sorted_token_ids.stream()).0 as *const i32,
            topk_weights_ptr,
            output.device_ptr(output.stream()).0 as *mut c_void,
            size_m as i32,
            size_n as i32,
            data_type_code,
            stream_raw,
        );
    }
    // Explicit drop just to make it obvious where the SyncOnDrop guard
    // finally fires — after the scatter launch has been queued. Rust
    // would drop this named binding at end of function anyway; without the
    // explicit drop the behaviour is identical.
    drop(topk_weights_ptr_holder);

    let output = candle::CudaStorage::wrap_cuda_slice(output, dev.clone());
    let output = Tensor::from_storage(
        candle::Storage::Cuda(output),
        (size_m, size_n),
        BackpropOp::none(),
        false,
    );
    Ok(output)
}
