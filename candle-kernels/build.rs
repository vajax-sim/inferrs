use std::env;
use std::path::PathBuf;

/// Default WMMA target arch when the user doesn't override `INFERRS_WMMA_ARCH`.
///
/// We pick `70` (Volta) so that compute_70 PTX embedded in the WMMA lib JITs
/// forward to every Tensor-Core arch in production today (V100, T4, A100,
/// L4, H100, B200, ...). PTX cannot JIT *backwards*, so picking 75 here
/// would silently break V100. Users who only care about Turing+ can set
/// `INFERRS_WMMA_ARCH=75` (or higher) for marginally better SASS.
const DEFAULT_WMMA_ARCH: usize = 70;

/// Top-level (non-MoE) kernels that go through the PTX build path.
const NON_MOE_KERNELS: &[&str] = &[
    "src/affine.cu",
    "src/binary.cu",
    "src/cast.cu",
    "src/conv.cu",
    "src/fill.cu",
    "src/flash_attn.cu",
    "src/indexing.cu",
    "src/quantized.cu",
    "src/reduce.cu",
    "src/sort.cu",
    "src/ternary.cu",
    "src/unary.cu",
];

fn main() {
    println!("cargo::rerun-if-changed=build.rs");
    println!("cargo::rerun-if-changed=src/compatibility.cuh");
    println!("cargo::rerun-if-changed=src/cuda_utils.cuh");
    println!("cargo::rerun-if-changed=src/binary_op_macros.cuh");
    println!("cargo::rerun-if-env-changed=INFERRS_WMMA_ARCH");
    println!("cargo::rerun-if-env-changed=CUDA_COMPUTE_CAP");

    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());

    let main_cc = detect_compute_cap();
    let wmma_arch: usize = env::var("INFERRS_WMMA_ARCH")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(DEFAULT_WMMA_ARCH);

    println!(
        "cargo:warning=candle-kernels: main_cc={} wmma_cc={}",
        main_cc
            .map(|n| n.to_string())
            .unwrap_or_else(|| "auto".into()),
        wmma_arch
    );

    let is_target_msvc = env::var("TARGET")
        .map(|t| t.contains("msvc"))
        .unwrap_or(false);

    // ------------------------------------------------------------------
    // Builder #1 — main PTX build for everything except src/moe/*.cu.
    // ------------------------------------------------------------------
    //
    // We override `kernel_paths` explicitly so the default `src/**/*.cu`
    // glob doesn't drag the MoE files in. The MoE kernels live in their
    // own static libs (Builders #2 and #3 below) and never need PTX.
    let ptx_path = out_dir.join("ptx.rs");
    let ptx_builder = bindgen_cuda::Builder::default()
        .arg("--expt-relaxed-constexpr")
        .arg("-std=c++17")
        .arg("-O3")
        .kernel_paths(NON_MOE_KERNELS.iter().map(PathBuf::from).collect());
    let bindings = ptx_builder.build_ptx().unwrap();
    bindings.write(&ptx_path).unwrap();

    // ------------------------------------------------------------------
    // Builder #2 — non-WMMA MoE static lib (libmoe_gguf.a).
    // ------------------------------------------------------------------
    //
    // `moe_gguf.cu` is a CUDA-core (no Tensor Core) GGUF MoE decode
    // kernel; it compiles cleanly on every arch we care about, so we
    // let bindgen_cuda use the default compute-cap detection.
    let mut gguf_builder = bindgen_cuda::Builder::default()
        .arg("--expt-relaxed-constexpr")
        .arg("-std=c++17")
        .arg("-O3")
        .kernel_paths(vec![PathBuf::from("src/moe/moe_gguf.cu")]);
    if is_target_msvc {
        gguf_builder = gguf_builder.arg("-D_USE_MATH_DEFINES");
    } else {
        gguf_builder = gguf_builder.arg("-Xcompiler").arg("-fPIC");
    }
    gguf_builder.build_lib(out_dir.join("libmoe_gguf.a"));

    // ------------------------------------------------------------------
    // Builder #3 — WMMA MoE static lib (libmoe_wmma.a).
    // ------------------------------------------------------------------
    //
    // `moe_wmma.cu` and `moe_wmma_gguf.cu` use the `nvcuda::wmma`
    // intrinsics which require compute capability >= 7.0 to *compile*,
    // even when targeting older hardware. We pin this Builder to a
    // WMMA-capable arch (default sm_70) regardless of the host GPU.
    //
    // The runtime gate in candle-nn/src/moe.rs checks
    // `CudaDevice::has_wmma()` before launching either of these kernels,
    // so on a Pascal box the host-side wrappers stay linked but the
    // sm_70 device code is never loaded.
    let mut wmma_builder = bindgen_cuda::Builder::default()
        .arg("--expt-relaxed-constexpr")
        .arg("-std=c++17")
        .arg("-O3")
        .compute_cap(wmma_arch)
        .kernel_paths(vec![
            PathBuf::from("src/moe/moe_wmma.cu"),
            PathBuf::from("src/moe/moe_wmma_gguf.cu"),
        ]);
    if is_target_msvc {
        wmma_builder = wmma_builder.arg("-D_USE_MATH_DEFINES");
    } else {
        wmma_builder = wmma_builder.arg("-Xcompiler").arg("-fPIC");
    }
    wmma_builder.build_lib(out_dir.join("libmoe_wmma.a"));

    // ------------------------------------------------------------------
    // Link both MoE libs plus the standard CUDA runtime.
    // ------------------------------------------------------------------
    println!("cargo:rustc-link-search={}", out_dir.display());
    println!("cargo:rustc-link-lib=moe_gguf");
    println!("cargo:rustc-link-lib=moe_wmma");
    println!("cargo:rustc-link-lib=dylib=cudart");
    if !is_target_msvc {
        println!("cargo:rustc-link-lib=stdc++");
    }
}

/// Detect the host GPU's CUDA compute capability for logging purposes.
///
/// `bindgen_cuda` does its own detection internally for Builders that don't
/// call `.compute_cap()`, so we only need this for the warning line.
fn detect_compute_cap() -> Option<u32> {
    if let Ok(v) = env::var("CUDA_COMPUTE_CAP") {
        if let Ok(n) = v.parse::<u32>() {
            return Some(n);
        }
    }
    let output = std::process::Command::new("nvidia-smi")
        .args(["--query-gpu=compute_cap", "--format=csv,noheader"])
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    let stdout = String::from_utf8(output.stdout).ok()?;
    let first = stdout.lines().next()?.trim();
    // e.g. "6.1" -> 61
    let digits: String = first.chars().filter(|c| c.is_ascii_digit()).collect();
    digits.parse::<u32>().ok()
}
