// Adapted from https://github.com/guoqingbao/attention.rs/blob/main/src/moe.rs
#[cfg(feature = "cuda")]
use candle::cuda_backend::kernels::ffi;
#[allow(unused_imports)]
use candle::quantized::{self, QTensor};
use candle::{Result, Tensor};

/// Returns `true` when the active CUDA device matches the build arch the
/// WMMA MoE kernels were compiled against (default sm_80, Ampere+).
///
/// The WMMA MoE kernels (`moe_wmma.cu`, `moe_wmma_gguf.cu`) instantiate
/// `nvcuda::wmma::fragment<..., nv_bfloat16, ...>` templates that require
/// compute capability 8.0 or higher (bf16 Tensor Cores landed in Ampere).
/// Half-precision WMMA works from Volta, but we need both, so the kernels
/// are compiled against sm_80 by default in `candle-kernels/build.rs`.
/// Because `bindgen_cuda::Builder::build_lib` emits SASS without a PTX
/// forward-JIT fallback, the cubin only runs on the exact arch it was
/// built for — and on older GPUs (Turing, Volta, Pascal, …) the driver
/// would fail at kernel-load time. This probe rejects those GPUs in Rust
/// before we ever try to launch the kernel, so the caller sees a clean
/// error instead of a segfault in the cubin loader.
///
/// We dlopen `libcuda.so` directly rather than going through `cudarc`
/// because the CUDA driver API exposes `cuDeviceGetAttribute` for compute
/// capability without requiring a context, and adapting cudarc's
/// version-pinned API surface here would be brittle.  The result is cached
/// in a `OnceLock` so the dlopen happens once per process.
#[cfg(feature = "cuda")]
fn has_wmma_support(_dev: &candle::CudaDevice) -> bool {
    use std::sync::OnceLock;
    static CACHED: OnceLock<bool> = OnceLock::new();
    *CACHED.get_or_init(probe_wmma_support)
}

#[cfg(feature = "cuda")]
fn probe_wmma_support() -> bool {
    use libloading::{Library, Symbol};
    type CuDeviceGetAttribute = unsafe extern "C" fn(*mut i32, i32, i32) -> i32;
    const CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR: i32 = 75;

    #[cfg(target_os = "windows")]
    let lib_names: &[&str] = &["nvcuda.dll"];
    #[cfg(not(target_os = "windows"))]
    let lib_names: &[&str] = &["libcuda.so.1", "libcuda.so"];

    let lib = match lib_names
        .iter()
        .find_map(|name| unsafe { Library::new(name).ok() })
    {
        Some(l) => l,
        None => return false,
    };

    let get_attr: Symbol<CuDeviceGetAttribute> =
        match unsafe { lib.get(b"cuDeviceGetAttribute\0") } {
            Ok(s) => s,
            Err(_) => return false,
        };

    // inferrs is single-GPU; ordinal 0 matches `is_cuda_uma_platform` in
    // inferrs/src/engine.rs.
    let ordinal: i32 = 0;
    let mut major: i32 = 0;
    let r = unsafe {
        get_attr(
            &mut major,
            CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,
            ordinal,
        )
    };
    if r != 0 {
        return false;
    }
    // Must match DEFAULT_WMMA_ARCH in candle-kernels/build.rs (Ampere+).
    major >= 8
}

#[cfg(feature = "cuda")]
pub fn moe_gemm(
    input: &Tensor,
    weights: &Tensor,
    topk_weights: &Option<Tensor>,
    sorted_token_ids: &Tensor,
    experts_ids: &Tensor,
    topk: usize,
    is_prefill: bool,
) -> Result<Tensor> {
    use candle::cuda_backend::cudarc::driver::DevicePtr;
    use candle::DType;
    use half::{bf16, f16};

    fn cuda_fwd<
        T: candle::cuda_backend::CudaDType + candle::cuda_backend::cudarc::driver::DeviceRepr,
    >(
        input: &Tensor,
        weights: &Tensor,
        topk_weights: &Option<Tensor>,
        sorted_token_ids: &Tensor,
        experts_ids: &Tensor,
        topk: usize,
        is_prefill: bool,
    ) -> Result<Tensor> {
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
        let dev = input.device().as_cuda_device()?;
        if !has_wmma_support(dev) {
            candle::bail!(
                "moe_gemm requires a CUDA device with bf16 Tensor Core \
                 support (compute capability >= 8.0, i.e. Ampere or newer). \
                 The active GPU is pre-Ampere; MoE inference is not yet \
                 supported on this hardware."
            );
        }
        let data_type = match input.dtype() {
            DType::F16 => 0,
            DType::BF16 => 1,
            _ => {
                candle::bail!("moe_gemm_wmma only accepts f16/bf16 inputs")
            }
        };

        let (input, _) = input.storage_and_layout();
        let input = match &*input {
            candle::Storage::Cuda(c) => c.as_cuda_slice::<T>()?,
            _ => candle::bail!("input must be a cuda tensor"),
        };

        let (weights, _) = weights.storage_and_layout();
        let weights = match &*weights {
            candle::Storage::Cuda(c) => c.as_cuda_slice::<T>()?,
            _ => candle::bail!("weight must be a cuda tensor"),
        };

        let (sorted_token_ids, _) = sorted_token_ids.storage_and_layout();
        let sorted_token_ids = match &*sorted_token_ids {
            candle::Storage::Cuda(c) => c.as_cuda_slice::<u32>()?,
            _ => candle::bail!("sorted_token_ids must be a cuda tensor"),
        };

        let (experts_ids, _) = experts_ids.storage_and_layout();
        let experts_ids = match &*experts_ids {
            candle::Storage::Cuda(c) => c.as_cuda_slice::<u32>()?,
            _ => candle::bail!("experts_ids must be a cuda tensor"),
        };

        let topk_weights_ptr = if let Some(topk_weights) = &topk_weights {
            let (topk_weights, _) = topk_weights.storage_and_layout();
            let topk_weights = match &*topk_weights {
                candle::Storage::Cuda(c) => c.as_cuda_slice::<f32>()?,
                _ => candle::bail!("topk_weights must be a cuda tensor"),
            };
            let weights_ptr = topk_weights.device_ptr(topk_weights.stream()).0 as *const f32;
            weights_ptr
        } else {
            std::ptr::null()
        };

        let output = unsafe { dev.alloc::<T>(size_m * size_n) }?;
        let expert_counts = unsafe { dev.alloc::<u32>(num_experts) }?;
        let expert_offsets = unsafe { dev.alloc::<u32>(num_experts + 1) }?;

        let stream = dev.cuda_stream().cu_stream() as i64;
        use core::ffi::c_void;

        unsafe {
            ffi::moe_gemm_wmma(
                input.device_ptr(input.stream()).0 as *const c_void, // [size_m, size_k]
                weights.device_ptr(weights.stream()).0 as *const c_void, // [num_experts, size_n, size_k]
                sorted_token_ids.device_ptr(sorted_token_ids.stream()).0 as *const i32,
                experts_ids.device_ptr(experts_ids.stream()).0 as *const i32,
                topk_weights_ptr,
                output.device_ptr(output.stream()).0 as *mut c_void, // [size_m, size_n]
                expert_counts.device_ptr(expert_counts.stream()).0 as *mut i32, // pre-allocated buffer [num_experts]
                expert_offsets.device_ptr(expert_offsets.stream()).0 as *mut i32, // pre-allocated buffer [num_experts + 1]
                num_experts as i32,
                topk as i32,
                size_m as i32,
                size_n as i32,
                size_k as i32,
                data_type as i32, // 0=float16, 1=bf16 (for input/output)
                is_prefill,
                stream,
            );
        }

        use candle::op::BackpropOp;
        let output = candle::CudaStorage::wrap_cuda_slice(output, dev.clone());
        let output = Tensor::from_storage(
            candle::Storage::Cuda(output),
            (size_m, size_n),
            BackpropOp::none(),
            false,
        );

        Ok(output)
    }

    match input.dtype() {
        DType::F16 => cuda_fwd::<f16>(
            input,
            weights,
            topk_weights,
            sorted_token_ids,
            experts_ids,
            topk,
            is_prefill,
        ),
        DType::BF16 => cuda_fwd::<bf16>(
            input,
            weights,
            topk_weights,
            sorted_token_ids,
            experts_ids,
            topk,
            is_prefill,
        ),
        _ => {
            candle::bail!("moe_gemm only accepts f16/bf16 inputs")
        }
    }
}

#[cfg(not(feature = "cuda"))]
pub fn moe_gemm(
    _: &Tensor,
    _: &Tensor,
    _: &Option<Tensor>,
    _: &Tensor,
    _: &Tensor,
    _: usize,
    _: bool,
) -> Result<Tensor> {
    candle::bail!("moe_gemm is only implemented for the cuda backend")
}

#[cfg(feature = "cuda")]
#[allow(clippy::too_many_arguments)]
pub fn moe_gemm_gguf(
    input: &Tensor,
    weights: &QTensor,
    topk_weights: &Option<Tensor>,
    sorted_token_ids: &Tensor,
    experts_ids: &Tensor,
    topk: usize,
    is_prefill: bool,
    dtype: candle::DType,
) -> Result<Tensor> {
    use candle::cuda_backend::cudarc::driver::DevicePtr;
    use candle::quantized::GgmlDType;
    use candle::DType;
    use half::{bf16, f16};

    #[allow(clippy::too_many_arguments)]
    fn cuda_fwd(
        input: &Tensor,
        weights: &QTensor,
        topk_weights: &Option<Tensor>,
        sorted_token_ids: &Tensor,
        experts_ids: &Tensor,
        topk: usize,
        is_prefill: bool,
        dtype: DType,
    ) -> Result<Tensor> {
        let (mut size_m, size_k) = input.dims2()?;
        if topk_weights.is_none() {
            size_m *= topk;
        }
        let (num_experts, size_n, size_k1) = weights.shape().dims3()?;
        assert!(
            size_k == size_k1,
            "input {:?} and weight {:?} last dim mismatch!",
            size_k,
            size_k1,
        );
        let dev = input.device().as_cuda_device()?;

        // Q8_0: 0, Q4K: 1, Q2K: 2, Q3k: 3,  Q5K: 4, Q6K: 5
        let gguf_dtype = match weights.dtype() {
            GgmlDType::Q8_0 => 0,
            GgmlDType::Q4K => 1,
            GgmlDType::Q2K => 2,
            GgmlDType::Q3K => 3,
            GgmlDType::Q5K => 4,
            GgmlDType::Q6K => 5,
            _ => {
                candle::bail!(
                    "moe_gemm_gguf `ISQ` only accept q2k, q3k, q4k, q5k, q6k or q8_0 weights!"
                )
            }
        };

        let weight_ptr = weights.device_ptr()?;

        let topk_weights_ptr = if let Some(topk_weights) = &topk_weights {
            let (topk_weights, _) = topk_weights.storage_and_layout();
            let topk_weights = match &*topk_weights {
                candle::Storage::Cuda(c) => c.as_cuda_slice::<f32>()?,
                _ => candle::bail!("topk_weights must be a cuda tensor"),
            };
            let w_ptr = topk_weights.device_ptr(topk_weights.stream()).0 as *const f32;
            w_ptr
        } else {
            std::ptr::null()
        };

        let (sorted_token_ids, _) = sorted_token_ids.storage_and_layout();
        let sorted_token_ids = match &*sorted_token_ids {
            candle::Storage::Cuda(c) => c.as_cuda_slice::<u32>()?,
            _ => candle::bail!("sorted_token_ids must be a cuda tensor"),
        };
        let (experts_ids, _) = experts_ids.storage_and_layout();
        let experts_ids = match &*experts_ids {
            candle::Storage::Cuda(c) => c.as_cuda_slice::<u32>()?,
            _ => candle::bail!("experts_ids must be a cuda tensor"),
        };

        let output = unsafe { dev.alloc::<f32>(size_m * size_n) }?;
        let stream = dev.cuda_stream().cu_stream() as i64;
        use candle::op::BackpropOp;
        use core::ffi::c_void;

        assert!(size_k % 8 == 0, "size_k must divisible by 8");
        if is_prefill && !has_wmma_support(dev) {
            candle::bail!(
                "moe_gemm_gguf prefill requires a CUDA device with bf16 \
                 Tensor Core support (compute capability >= 8.0, i.e. \
                 Ampere or newer). The active GPU is pre-Ampere; GGUF MoE \
                 prefill is not yet supported on this hardware (decode \
                 still works)."
            );
        }
        unsafe {
            if is_prefill {
                let input = input.to_dtype(dtype)?;
                let (input, _) = input.storage_and_layout();
                let (input_ptr, input_dtype) = match &*input {
                    candle::Storage::Cuda(c) => {
                        if dtype == DType::F16 {
                            let c = c.as_cuda_slice::<f16>()?;
                            (c.device_ptr(c.stream()).0 as *const c_void, 0)
                        } else {
                            let c = c.as_cuda_slice::<bf16>()?;
                            (c.device_ptr(c.stream()).0 as *const c_void, 1)
                        }
                    }
                    _ => candle::bail!("input must be a cuda tensor"),
                };
                ffi::moe_gemm_gguf_prefill(
                    input_ptr,  // [size_m or size_m/topk, size_k]
                    weight_ptr, // [num_experts, size_n, size_k]
                    sorted_token_ids.device_ptr(sorted_token_ids.stream()).0 as *const i32,
                    experts_ids.device_ptr(experts_ids.stream()).0 as *const i32,
                    topk_weights_ptr,
                    output.device_ptr(output.stream()).0 as *mut c_void, // [size_m, size_n]
                    num_experts as i32,
                    topk as i32,
                    size_m as i32,
                    size_n as i32,
                    size_k as i32,
                    input_dtype,
                    gguf_dtype as i32, // Q8_0: 0, Q4K: 1, Q2K: 2, Q3k: 3,  Q5K: 4, Q6K: 5 (for weight)
                    stream,
                );
            } else {
                let (input, _) = input.storage_and_layout();
                let input = match &*input {
                    candle::Storage::Cuda(c) => c.as_cuda_slice::<f32>()?,
                    _ => candle::bail!("input must be a cuda tensor"),
                };

                ffi::moe_gemm_gguf(
                    input.device_ptr(input.stream()).0 as *const f32, // [size_m or size_m/topk, size_k]
                    weight_ptr as *const c_void, // [num_experts, size_n, size_k]
                    sorted_token_ids.device_ptr(sorted_token_ids.stream()).0 as *const i32,
                    experts_ids.device_ptr(experts_ids.stream()).0 as *const i32,
                    topk_weights_ptr,
                    output.device_ptr(output.stream()).0 as *mut c_void, // [size_m, size_n]
                    num_experts as i32,
                    topk as i32,
                    size_m as i32,
                    size_n as i32,
                    size_k as i32,
                    gguf_dtype as i32, // Q8_0: 0, Q4K: 1, Q2K: 2, Q3k: 3,  Q5K: 4, Q6K: 5 (for weight)
                    stream,
                );
            }
        }

        let output = candle::CudaStorage::wrap_cuda_slice(output, dev.clone());
        let output = Tensor::from_storage(
            candle::Storage::Cuda(output),
            (size_m, size_n),
            BackpropOp::none(),
            false,
        );

        Ok(output)
    }

    match input.dtype() {
        DType::F32 => cuda_fwd(
            input,
            weights,
            topk_weights,
            sorted_token_ids,
            experts_ids,
            topk,
            is_prefill,
            dtype,
        ),
        _ => {
            candle::bail!("moe_gemm_gguf only accepts f32 inputs")
        }
    }
}

#[cfg(not(feature = "cuda"))]
#[allow(clippy::too_many_arguments)]
pub fn moe_gemm_gguf(
    _: &Tensor,
    _: &QTensor,
    _: &Option<Tensor>,
    _: &Tensor,
    _: &Tensor,
    _: usize,
    _: bool,
    _: candle::DType,
) -> Result<Tensor> {
    candle::bail!("moe_gemm_gguf is only implemented for the cuda backend")
}
