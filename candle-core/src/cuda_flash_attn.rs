/// CUDA Flash Attention decode for BF16 GQA tensors.
///
/// Dispatches `flash_attn_decode_bf16_dD` from candle-kernels/flash_attn.cu.
///
/// Q:   `[1, n_q_heads, 1, head_dim]`  BF16
/// K/V: `[1, n_kv_heads, kv_len, head_dim]`  BF16
/// Out: `[1, n_q_heads, 1, head_dim]`  F32
use crate::{op::BackpropOp, DType, Result, Storage, Tensor};

pub fn flash_attn_decode_cuda(q: &Tensor, k: &Tensor, v: &Tensor, scale: f32) -> Result<Tensor> {
    use candle_kernels as kernels;
    use cudarc::driver::PushKernelArg;

    let cuda_dev = match q.device() {
        crate::Device::Cuda(d) => d.clone(),
        _ => crate::bail!("flash_attn_decode_cuda requires CUDA device"),
    };

    let (_, n_q, q_len, head_dim) = q.dims4()?;
    let (_, n_kv, kv_len, _) = k.dims4()?;

    if q_len != 1 {
        crate::bail!("flash_attn_decode_cuda: q_len={} must be 1", q_len);
    }
    if n_q % n_kv != 0 {
        crate::bail!("n_q={} not divisible by n_kv={}", n_q, n_kv);
    }
    if q.dtype() != DType::BF16 {
        crate::bail!(
            "flash_attn_decode_cuda: expected BF16 q, got {:?}",
            q.dtype()
        );
    }

    let kernel_name = match head_dim {
        64 => "flash_attn_decode_bf16_d64",
        128 => "flash_attn_decode_bf16_d128",
        256 => "flash_attn_decode_bf16_d256",
        512 => "flash_attn_decode_bf16_d512",
        _ => crate::bail!("flash_attn_decode_cuda: unsupported head_dim={}", head_dim),
    };

    let n_kv_groups = (n_q / n_kv) as i32;

    // Ensure contiguous layout.
    let q_c = q.contiguous()?;
    let k_c = k.contiguous()?;
    let v_c = v.contiguous()?;

    // Extract BF16 CUDA slices.  We hold the read-guards for the duration of the launch.
    let (q_stor, q_lay) = q_c.storage_and_layout();
    let (q_o1, q_o2) = q_lay
        .contiguous_offsets()
        .ok_or_else(|| crate::Error::msg("q not contiguous after contiguous()"))?;
    let q_slice = match &*q_stor {
        Storage::Cuda(cs) => cs.as_cuda_slice::<half::bf16>()?.slice(q_o1..q_o2),
        _ => crate::bail!("expected Cuda storage for q"),
    };

    let (k_stor, k_lay) = k_c.storage_and_layout();
    let (k_o1, k_o2) = k_lay
        .contiguous_offsets()
        .ok_or_else(|| crate::Error::msg("k not contiguous"))?;
    let k_slice = match &*k_stor {
        Storage::Cuda(cs) => cs.as_cuda_slice::<half::bf16>()?.slice(k_o1..k_o2),
        _ => crate::bail!("expected Cuda storage for k"),
    };

    let (v_stor, v_lay) = v_c.storage_and_layout();
    let (v_o1, v_o2) = v_lay
        .contiguous_offsets()
        .ok_or_else(|| crate::Error::msg("v not contiguous"))?;
    let v_slice = match &*v_stor {
        Storage::Cuda(cs) => cs.as_cuda_slice::<half::bf16>()?.slice(v_o1..v_o2),
        _ => crate::bail!("expected Cuda storage for v"),
    };

    // Allocate F32 output buffer: n_q_heads * head_dim elements.
    let out_elems = n_q * head_dim;
    let out_buf = unsafe {
        cuda_dev
            .alloc::<f32>(out_elems)
            .map_err(|e| crate::Error::Cuda(Box::new(e)))?
    };

    // Dynamic shared memory: one float per warp (for partial dot-product sums).
    let n_warps = ((head_dim as u32) + 31) / 32;
    let shared_bytes = n_warps * std::mem::size_of::<f32>() as u32;

    let func = cuda_dev
        .get_or_load_func(kernel_name, &kernels::FLASH_ATTN)
        .map_err(|e| crate::Error::Cuda(Box::new(e)))?;

    let cfg = cudarc::driver::LaunchConfig {
        grid_dim: (n_q as u32, 1, 1),
        block_dim: (head_dim as u32, 1, 1),
        shared_mem_bytes: shared_bytes,
    };

    {
        let kv_len_i = kv_len as i32;
        let mut b = func.builder();
        b.arg(&q_slice);
        b.arg(&k_slice);
        b.arg(&v_slice);
        b.arg(&out_buf);
        b.arg(&n_kv_groups);
        b.arg(&kv_len_i);
        b.arg(&scale);
        unsafe { b.launch(cfg) }.map_err(|e| crate::Error::Cuda(Box::new(e)))?;
    }

    drop(q_stor);
    drop(k_stor);
    drop(v_stor);

    // Build output tensor [1, n_q_heads, 1, head_dim] in F32.
    let out_cs = crate::CudaStorage::wrap_cuda_slice(out_buf, cuda_dev);
    let shape = crate::Shape::from_dims(&[1usize, n_q, 1, head_dim]);
    Ok(Tensor::from_storage(
        Storage::Cuda(out_cs),
        shape,
        BackpropOp::none(),
        false,
    ))
}
