//! Token sampling: temperature, top-k, top-p, repetition penalty.

use anyhow::Result;
use candle_core::{DType, Tensor};

/// Sampling parameters for a generation request.
#[derive(Debug, Clone)]
pub struct SamplingParams {
    pub temperature: f64,
    pub top_p: f64,
    pub top_k: usize,
    pub repetition_penalty: f64,
    pub max_tokens: usize,
}

impl Default for SamplingParams {
    fn default() -> Self {
        Self {
            temperature: 0.7,
            top_p: 0.9,
            top_k: 50,
            repetition_penalty: 1.0,
            max_tokens: 2048,
        }
    }
}

const SAMPLING_EPS: f64 = 1e-5;

/// Sample a token from logits using the given parameters.
pub fn sample_token(
    logits: &Tensor,
    params: &SamplingParams,
    previous_tokens: &[u32],
) -> Result<u32> {
    // logits shape: (1, 1, vocab_size) or (1, vocab_size) or (vocab_size,)
    // Flatten to 1D
    let logits = logits.squeeze(0)?;
    let logits = if logits.dims().len() > 1 {
        logits.squeeze(0)?
    } else {
        logits
    };
    let logits = logits.to_dtype(DType::F32)?;

    // Apply repetition penalty
    let logits = if params.repetition_penalty != 1.0 && !previous_tokens.is_empty() {
        apply_repetition_penalty(&logits, previous_tokens, params.repetition_penalty)?
    } else {
        logits
    };

    // Greedy sampling if temperature is very low
    if params.temperature < SAMPLING_EPS {
        let token_id = logits.argmax(0)?.to_scalar::<u32>()?;
        return Ok(token_id);
    }

    // Apply temperature
    let logits = (&logits / params.temperature)?;

    // Convert to probabilities
    let probs = candle_nn::ops::softmax_last_dim(&logits)?;
    let probs_vec: Vec<f32> = probs.to_vec1()?;

    // Apply top-k filtering
    let mut indexed_probs: Vec<(usize, f32)> = probs_vec.iter().copied().enumerate().collect();
    indexed_probs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    // Top-k
    if params.top_k > 0 && params.top_k < indexed_probs.len() {
        indexed_probs.truncate(params.top_k);
    }

    // Top-p (nucleus sampling)
    if params.top_p < 1.0 {
        let mut cumsum = 0.0f32;
        let mut cutoff_idx = indexed_probs.len();
        for (i, &(_, p)) in indexed_probs.iter().enumerate() {
            cumsum += p;
            if cumsum >= params.top_p as f32 {
                cutoff_idx = i + 1;
                break;
            }
        }
        indexed_probs.truncate(cutoff_idx);
    }

    // Renormalize
    let sum: f32 = indexed_probs.iter().map(|&(_, p)| p).sum();
    if sum <= 0.0 {
        // Fallback to argmax
        return Ok(indexed_probs
            .first()
            .map(|&(idx, _)| idx as u32)
            .unwrap_or(0));
    }

    // Sample from the filtered distribution
    let mut rng_val: f32 = rand_f32();
    for &(idx, prob) in &indexed_probs {
        let normalized = prob / sum;
        if rng_val < normalized {
            return Ok(idx as u32);
        }
        rng_val -= normalized;
    }

    // Fallback
    Ok(indexed_probs
        .last()
        .map(|&(idx, _)| idx as u32)
        .unwrap_or(0))
}

fn apply_repetition_penalty(
    logits: &Tensor,
    previous_tokens: &[u32],
    penalty: f64,
) -> Result<Tensor> {
    let logits_vec: Vec<f32> = logits.to_vec1()?;
    let mut modified = logits_vec;

    for &token_id in previous_tokens {
        let idx = token_id as usize;
        if idx < modified.len() {
            let score = modified[idx];
            // If score > 0, divide by penalty; if score < 0, multiply by penalty
            modified[idx] = if score > 0.0 {
                score / penalty as f32
            } else {
                score * penalty as f32
            };
        }
    }

    Ok(Tensor::from_vec(modified, logits.shape(), logits.device())?)
}

/// Simple random float in [0, 1) using thread-local RNG.
fn rand_f32() -> f32 {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};
    use std::time::SystemTime;

    // Use a thread-local counter + time for randomness
    thread_local! {
        static COUNTER: std::cell::Cell<u64> = const { std::cell::Cell::new(0) };
    }

    COUNTER.with(|c| {
        let count = c.get().wrapping_add(1);
        c.set(count);

        let mut hasher = DefaultHasher::new();
        count.hash(&mut hasher);
        SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos()
            .hash(&mut hasher);
        std::thread::current().id().hash(&mut hasher);

        let hash = hasher.finish();
        (hash as f32) / (u64::MAX as f32)
    })
}
