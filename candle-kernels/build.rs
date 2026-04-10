use std::env;
use std::path::PathBuf;

fn main() {
    println!("cargo::rerun-if-changed=build.rs");
    println!("cargo::rerun-if-changed=src/compatibility.cuh");
    println!("cargo::rerun-if-changed=src/cuda_utils.cuh");
    println!("cargo::rerun-if-changed=src/binary_op_macros.cuh");

    // WMMA (Tensor Core) intrinsics require SM 7.0+ (Volta).  When targeting
    // older architectures (e.g. Pascal SM 6.x), pass -DINFERRS_NO_WMMA so the
    // WMMA .cu files compile to no-op stubs instead of hitting undefined
    // identifier errors for the nvcuda::wmma namespace.
    //
    // bindgen_cuda auto-detects the GPU compute capability via nvidia-smi and
    // sets CUDA_COMPUTE_CAP for nested builder invocations, but during the
    // *first* build_ptx() call the env var is not yet populated.  Replicate
    // the same detection here so both the PTX build and the lib build agree.
    let compute_cap: u32 = detect_compute_cap().unwrap_or(80);
    let has_wmma = compute_cap >= 70;
    println!(
        "cargo:warning=candle-kernels: CUDA_COMPUTE_CAP={} has_wmma={}",
        compute_cap, has_wmma
    );

    // Build for PTX
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    let ptx_path = out_dir.join("ptx.rs");
    let mut builder = bindgen_cuda::Builder::default()
        .arg("--expt-relaxed-constexpr")
        .arg("-std=c++17")
        .arg("-O3");
    if !has_wmma {
        builder = builder.arg("-DINFERRS_NO_WMMA");
    }
    let bindings = builder.build_ptx().unwrap();
    bindings.write(&ptx_path).unwrap();

    // Remove unwanted MOE PTX constants from ptx.rs
    remove_lines(&ptx_path, &["MOE_GGUF", "MOE_WMMA", "MOE_WMMA_GGUF"]);

    let mut moe_builder = bindgen_cuda::Builder::default()
        .arg("--expt-relaxed-constexpr")
        .arg("-std=c++17")
        .arg("-O3");

    if !has_wmma {
        moe_builder = moe_builder.arg("-DINFERRS_NO_WMMA");
    }

    // Build for FFI binding (must use custom bindgen_cuda, which supports simutanously build PTX and lib)
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    let mut is_target_msvc = false;
    if let Ok(target) = std::env::var("TARGET") {
        if target.contains("msvc") {
            is_target_msvc = true;
            moe_builder = moe_builder.arg("-D_USE_MATH_DEFINES");
        }
    }

    if !is_target_msvc {
        moe_builder = moe_builder.arg("-Xcompiler").arg("-fPIC");
    }

    let moe_builder = moe_builder.kernel_paths(vec![
        "src/moe/moe_gguf.cu",
        "src/moe/moe_wmma.cu",
        "src/moe/moe_wmma_gguf.cu",
    ]);
    moe_builder.build_lib(out_dir.join("libmoe.a"));
    println!("cargo:rustc-link-search={}", out_dir.display());
    println!("cargo:rustc-link-lib=moe");
    println!("cargo:rustc-link-lib=dylib=cudart");
    if !is_target_msvc {
        println!("cargo:rustc-link-lib=stdc++");
    }
}

/// Detect the CUDA compute capability the kernels will be built for.
///
/// Mirrors what `bindgen_cuda` does internally: honour CUDA_COMPUTE_CAP if
/// set, otherwise query nvidia-smi for the GPU's native capability.
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

fn remove_lines<P: AsRef<std::path::Path>>(file: P, patterns: &[&str]) {
    let content = std::fs::read_to_string(&file).unwrap();
    let filtered = content
        .lines()
        .filter(|line| !patterns.iter().any(|p| line.contains(p)))
        .collect::<Vec<_>>()
        .join("\n");
    std::fs::write(file, filtered).unwrap();
}
