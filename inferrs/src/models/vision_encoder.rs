//! Gemma 4 vision encoder (SigLIP2 ViT-B/16).
//!
//! Architecture:
//!
//! - `PatchEmbedder`: linear patch projection + 2D factored position embedding table
//! - N × `VisionEncoderLayer`:
//!   - input_layernorm → SelfAttention (q/k-norm, 2D RoPE, ClipLinear) → post_attention_layernorm
//!   - pre_feedforward_layernorm → GeGLU MLP (ClipLinear) → post_feedforward_layernorm
//! - `VisionPooler`: position-aware 2D average pooling → scale by sqrt(hidden_size)
//! - `embed_proj` linear: hidden_size → lm_hidden_size
//!
//! Weight prefixes:
//!   - Vision tower: `model.vision_tower.*`
//!   - LM projection: `model.embed_vision.embedding_projection`
//!
//! Input: pre-patchified `pixel_values` of shape `[N_patches, patch_size² × 3]`
//!   (f32, values in [0, 1]).  The PatchEmbedder applies `2*(x-0.5)` internally.
//! Input: `pixel_position_ids` of shape `[N_patches, 2]` (i32, x/y patch coordinates).
//! Output: `[n_soft_tokens, lm_hidden_size]` — ready for injection into the LM.

use anyhow::{Context, Result};
use candle_core::{DType, Device, Module, Tensor, D};
use candle_nn::{linear_no_bias, rms_norm, Linear, RmsNorm, VarBuilder};

use crate::config::VisionConfig;
use crate::models::audio_encoder::ClipLinear;

// ---------------------------------------------------------------------------
// Patch embedder
// ---------------------------------------------------------------------------

/// Projects pre-patchified pixel values to hidden_size and adds 2D position
/// embeddings from a factored position embedding table.
///
/// Weight layout (under `patch_embedder`):
///   `input_proj.weight`         [hidden_size, patch_size² × 3]  (no bias)
///   `position_embedding_table`  [2, position_embedding_size, hidden_size]
///
/// Note: the scale factor `2*(x-0.5)` is applied inside this module,
/// so callers pass raw [0, 1] float values.
struct PatchEmbedder {
    /// Linear: (patch_size² × 3) → hidden_size, no bias.
    input_proj: Linear,
    /// Factored 2D position embedding table: [2, pos_emb_size, hidden_size].
    position_embedding_table: Tensor,
    pos_emb_size: usize,
}

impl PatchEmbedder {
    fn load(vb: VarBuilder, cfg: &VisionConfig) -> Result<Self> {
        let patch_pixels = cfg.patch_size * cfg.patch_size * 3;
        let input_proj = linear_no_bias(patch_pixels, cfg.hidden_size, vb.pp("input_proj"))?;
        let position_embedding_table = vb.get(
            (2, cfg.position_embedding_size, cfg.hidden_size),
            "position_embedding_table",
        )?;
        Ok(Self {
            input_proj,
            position_embedding_table,
            pos_emb_size: cfg.position_embedding_size,
        })
    }

    /// Forward pass.
    ///
    /// `pixel_values`:     `[N_patches, patch_pixels]` — values in [0, 1].
    /// `position_ids`:     `[N_patches, 2]`            — (x, y) coords, i64.
    /// `padding_mask`:     `[N_patches]`               — true for padding patches.
    ///
    /// Returns `[N_patches, hidden_size]`.
    fn forward(
        &self,
        pixel_values: &Tensor,
        position_ids: &Tensor,
        padding_mask: &Tensor,
    ) -> Result<Tensor> {
        // Scale [0,1] → [-1,1] (2*x - 1).
        let pv = ((pixel_values * 2.0)? - 1.0)?;

        // Project patches: [N, patch_pixels] → [N, hidden_size].
        let hidden = self.input_proj.forward(&pv)?;

        // 2D position embeddings.
        // position_ids: [N, 2] i64 with values in [0, pos_emb_size) or -1 for padding.
        let pos_emb = self.position_embedding(position_ids, padding_mask)?;

        Ok((hidden + pos_emb)?)
    }

    /// Compute factored 2D position embeddings.
    ///
    /// For each patch at (x, y):
    ///   pos_emb = table[0, x] + table[1, y]
    /// Padding patches (position == -1) get zero embeddings.
    fn position_embedding(&self, position_ids: &Tensor, padding_mask: &Tensor) -> Result<Tensor> {
        // Clamp -1 padding values to 0 before indexing.
        let clamped = position_ids.clamp(0i64, (self.pos_emb_size as i64) - 1)?;

        // x coords: [N] — select from table[0, :, :] → [N, hidden]
        // `.contiguous()` is required because Metal's index_select kernel does
        // not accept non-contiguous index tensors, and narrow+squeeze leaves the
        // index tensor non-contiguous on Metal.
        let xs = clamped.narrow(1, 0, 1)?.squeeze(1)?.contiguous()?; // [N]
        let ys = clamped.narrow(1, 1, 1)?.squeeze(1)?.contiguous()?; // [N]

        // `.contiguous()` is required: narrow+squeeze leaves the table slices
        // non-contiguous on Metal, which causes index_select to return wrong shapes.
        let table_x = self
            .position_embedding_table
            .narrow(0, 0, 1)?
            .squeeze(0)?
            .contiguous()?; // [pos, hidden]
        let table_y = self
            .position_embedding_table
            .narrow(0, 1, 1)?
            .squeeze(0)?
            .contiguous()?; // [pos, hidden]

        let emb_x = table_x.index_select(&xs, 0)?; // [N, hidden]
        let emb_y = table_y.index_select(&ys, 0)?; // [N, hidden]

        let pos_emb = (emb_x + emb_y)?;

        // Zero out padding patches.
        // padding_mask: [N] bool-like (1.0 = padding, 0.0 = real)
        let mask = padding_mask.unsqueeze(1)?; // [N, 1]
                                               // where mask==1.0 → 0, else pos_emb
                                               // masked = pos_emb * (1 - mask)
        let ones = Tensor::ones_like(&mask)?;
        let inv_mask = (ones - mask)?;
        let masked = inv_mask.broadcast_mul(&pos_emb)?;
        Ok(masked)
    }
}

// ---------------------------------------------------------------------------
// Vision attention (full bidirectional, 2D RoPE, Q/K RMSNorm)
// ---------------------------------------------------------------------------

struct VisionAttention {
    q_proj: ClipLinear,
    k_proj: ClipLinear,
    v_proj: ClipLinear,
    o_proj: ClipLinear,
    q_norm: RmsNorm,
    k_norm: RmsNorm,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    /// RoPE inverse frequencies: [head_dim/4] (half used per spatial dim).
    inv_freq: Tensor,
}

impl VisionAttention {
    fn load(vb: VarBuilder, cfg: &VisionConfig) -> Result<Self> {
        let h = cfg.hidden_size;
        let nh = cfg.num_attention_heads;
        let nkv = cfg.num_key_value_heads;
        let hd = cfg.head_dim;

        let attn_vb = vb.pp("self_attn");
        let q_proj = ClipLinear::load(attn_vb.pp("q_proj"), h, nh * hd)?;
        let k_proj = ClipLinear::load(attn_vb.pp("k_proj"), h, nkv * hd)?;
        let v_proj = ClipLinear::load(attn_vb.pp("v_proj"), h, nkv * hd)?;
        let o_proj = ClipLinear::load(attn_vb.pp("o_proj"), nh * hd, h)?;

        let q_norm = rms_norm(hd, cfg.rms_norm_eps, attn_vb.pp("q_norm"))?;
        let k_norm = rms_norm(hd, cfg.rms_norm_eps, attn_vb.pp("k_norm"))?;

        // Build RoPE inv_freq on CPU.
        // Vision uses head_dim/2 channels per spatial dim; each dim gets head_dim/4 inv_freqs.
        let spatial_dim = hd / 2; // channels per spatial dimension
        let n_freq = spatial_dim / 2;
        let theta = cfg.rope_theta;
        let inv_freq = build_inv_freq(n_freq, theta, &Device::Cpu)?;

        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            q_norm,
            k_norm,
            num_heads: nh,
            num_kv_heads: nkv,
            head_dim: hd,
            inv_freq,
        })
    }

    /// Forward.
    ///
    /// `hidden_states`: `[N, hidden_size]` — no batch dim (single image).
    /// `position_ids`:  `[N, 2]`          — (x, y) patch coords, i64.
    ///
    /// Returns `[N, hidden_size]`.
    fn forward(&self, hidden_states: &Tensor, position_ids: &Tensor) -> Result<Tensor> {
        let (n, _h) = hidden_states.dims2()?;
        let nh = self.num_heads;
        let nkv = self.num_kv_heads;
        let hd = self.head_dim;

        // Project.
        let q = self.q_proj.forward(hidden_states)?; // [N, nh*hd]
        let k = self.k_proj.forward(hidden_states)?; // [N, nkv*hd]
        let v = self.v_proj.forward(hidden_states)?; // [N, nkv*hd]

        // Reshape to [N, heads, head_dim].
        let q = q.reshape((n, nh, hd))?;
        let k = k.reshape((n, nkv, hd))?;
        let v = v.reshape((n, nkv, hd))?;

        // Per-head RMSNorm on Q and K.
        let q = self.q_norm.forward(&q)?;
        let k = self.k_norm.forward(&k)?;

        // 2D RoPE: apply separately to x and y halves.
        let q = apply_2d_rope(&q, position_ids, &self.inv_freq)?; // [N, nh, hd]
        let k = apply_2d_rope(&k, position_ids, &self.inv_freq)?; // [N, nkv, hd]

        // Transpose for attention: [heads, N, hd].
        // `.contiguous()` is required: Metal matmul kernels require contiguous inputs,
        // and permute produces a non-contiguous view.
        let q = q.permute((1, 0, 2))?.contiguous()?;
        let k = k.permute((1, 0, 2))?.contiguous()?;
        let v = v.permute((1, 0, 2))?.contiguous()?;

        // GQA: expand K/V to match Q heads if needed.
        let k = if nkv != nh {
            repeat_kv(k, nh / nkv)?
        } else {
            k
        };
        let v = if nkv != nh {
            repeat_kv(v, nh / nkv)?
        } else {
            v
        };

        // Scaled dot-product attention (no mask — bidirectional vision encoder).
        let scale = (hd as f64).sqrt();
        let attn = (q.matmul(&k.transpose(1, 2)?)? / scale)?; // [nh, N, N]
        let attn = candle_nn::ops::softmax_last_dim(&attn)?;
        let out = attn.matmul(&v)?; // [nh, N, hd]

        // Merge heads: [N, nh*hd].
        let out = out.permute((1, 0, 2))?.reshape((n, nh * hd))?;

        // Output projection.
        self.o_proj.forward(&out)
    }
}

// ---------------------------------------------------------------------------
// Vision MLP (GeGLU with ClipLinear)
// ---------------------------------------------------------------------------

struct VisionMlp {
    gate_proj: ClipLinear,
    up_proj: ClipLinear,
    down_proj: ClipLinear,
}

impl VisionMlp {
    fn load(vb: VarBuilder, cfg: &VisionConfig) -> Result<Self> {
        let mlp_vb = vb.pp("mlp");
        let gate_proj = ClipLinear::load(
            mlp_vb.pp("gate_proj"),
            cfg.hidden_size,
            cfg.intermediate_size,
        )?;
        let up_proj =
            ClipLinear::load(mlp_vb.pp("up_proj"), cfg.hidden_size, cfg.intermediate_size)?;
        let down_proj = ClipLinear::load(
            mlp_vb.pp("down_proj"),
            cfg.intermediate_size,
            cfg.hidden_size,
        )?;
        Ok(Self {
            gate_proj,
            up_proj,
            down_proj,
        })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let gate = self.gate_proj.forward(xs)?.gelu_erf()?;
        let up = self.up_proj.forward(xs)?;
        let mid = (gate * up)?;
        self.down_proj.forward(&mid)
    }
}

// ---------------------------------------------------------------------------
// Vision encoder layer
// ---------------------------------------------------------------------------

struct VisionEncoderLayer {
    input_layernorm: RmsNorm,
    post_attention_layernorm: RmsNorm,
    pre_feedforward_layernorm: RmsNorm,
    post_feedforward_layernorm: RmsNorm,
    self_attn: VisionAttention,
    mlp: VisionMlp,
}

impl VisionEncoderLayer {
    fn load(vb: VarBuilder, cfg: &VisionConfig) -> Result<Self> {
        let input_layernorm =
            rms_norm(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("input_layernorm"))?;
        let post_attention_layernorm = rms_norm(
            cfg.hidden_size,
            cfg.rms_norm_eps,
            vb.pp("post_attention_layernorm"),
        )?;
        let pre_feedforward_layernorm = rms_norm(
            cfg.hidden_size,
            cfg.rms_norm_eps,
            vb.pp("pre_feedforward_layernorm"),
        )?;
        let post_feedforward_layernorm = rms_norm(
            cfg.hidden_size,
            cfg.rms_norm_eps,
            vb.pp("post_feedforward_layernorm"),
        )?;
        let self_attn = VisionAttention::load(vb.clone(), cfg)?;
        let mlp = VisionMlp::load(vb.clone(), cfg)?;
        Ok(Self {
            input_layernorm,
            post_attention_layernorm,
            pre_feedforward_layernorm,
            post_feedforward_layernorm,
            self_attn,
            mlp,
        })
    }

    fn forward(&self, xs: &Tensor, position_ids: &Tensor) -> Result<Tensor> {
        // Attention sub-layer.
        let residual = xs;
        let h = self.input_layernorm.forward(xs)?;
        let h = self.self_attn.forward(&h, position_ids)?;
        let h = self.post_attention_layernorm.forward(&h)?;
        let xs = (residual + h)?;

        // MLP sub-layer.
        let residual = &xs;
        let h = self.pre_feedforward_layernorm.forward(&xs)?;
        let h = self.mlp.forward(&h)?;
        let h = self.post_feedforward_layernorm.forward(&h)?;
        (residual + h).map_err(Into::into)
    }
}

// ---------------------------------------------------------------------------
// VisionPooler: position-aware 2D average pooling
// ---------------------------------------------------------------------------

struct VisionPooler {
    root_hidden_size: f64,
}

impl VisionPooler {
    fn new(cfg: &VisionConfig) -> Self {
        Self {
            root_hidden_size: (cfg.hidden_size as f64).sqrt(),
        }
    }

    /// Pool `hidden_states [N_patches, hidden]` to `[output_length, hidden]`.
    ///
    /// Uses the `position_ids` to assign patches to output bins.
    /// Padding patches (position_ids == -1) contribute zero.
    ///
    /// `position_ids`: `[N_patches, 2]` i64 — x/y patch coordinates.
    /// `padding_mask`: `[N_patches]` bool tensor (1 = padding).
    fn forward(
        &self,
        hidden_states: &Tensor,
        position_ids: &Tensor,
        padding_mask: &Tensor,
        output_length: usize,
        kernel: usize,
    ) -> Result<Tensor> {
        let (_n, h) = hidden_states.dims2()?;
        let dev = hidden_states.device();
        let dtype = hidden_states.dtype();

        // Zero out padding patches.
        // broadcast_mul requires the broadcaster ([N,1]) on the left; use
        // explicit broadcast_mul rather than * so Metal picks the right operand order.
        let pmask = padding_mask.unsqueeze(1)?.to_dtype(dtype)?; // [N,1]
        let inv_pmask = (Tensor::ones_like(&pmask)? - &pmask)?; // [N,1]
        let hs = inv_pmask.broadcast_mul(hidden_states)?; // [N, h]

        // Compute output bin index for each patch.
        // max_x = max(position_ids[:, 0]) + 1
        let pos_x = position_ids.narrow(1, 0, 1)?.squeeze(1)?; // [N] i64
        let pos_y = position_ids.narrow(1, 1, 1)?.squeeze(1)?; // [N] i64

        // Clamp negative (padding) to 0 — they won't contribute due to zero masking above.
        let cx = pos_x.clamp(0i64, i64::MAX)?;
        let cy = pos_y.clamp(0i64, i64::MAX)?;

        // Kernel pooling: divide each coord by kernel to get bin index.
        // bin_x = x / kernel,  bin_y = y / kernel
        let bx = (&cx / kernel as f64)?;
        let by = (&cy / kernel as f64)?;

        // Figure out grid width (num output cols) from max bin_x + 1.
        let bx_max = bx.max(0)?.to_scalar::<i64>()? as usize + 1;
        let bin_idx = (by * bx_max as f64 + bx)?; // [N] i64 linear bin index

        // Scatter-add: accumulate each patch into its bin.
        // Also count how many patches fall in each bin (for averaging).
        let bin_idx_usize: Vec<i64> = bin_idx.to_vec1::<i64>()?;
        let hs_vec: Vec<f32> = hs.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;

        let mut acc = vec![0.0f32; output_length * h];
        let mut counts = vec![0u32; output_length];

        for (i, &bin) in bin_idx_usize.iter().enumerate() {
            let bin = bin as usize;
            if bin < output_length {
                for d in 0..h {
                    acc[bin * h + d] += hs_vec[i * h + d];
                }
                counts[bin] += 1;
            }
        }

        // Divide by count (average) — non-empty bins only.
        for bin in 0..output_length {
            let c = counts[bin];
            if c > 1 {
                let scale = 1.0 / c as f32;
                for d in 0..h {
                    acc[bin * h + d] *= scale;
                }
            }
        }

        let pooled = Tensor::from_vec(acc, (output_length, h), dev)?.to_dtype(dtype)?;

        // Scale by sqrt(hidden_size).
        Ok((pooled * self.root_hidden_size)?)
    }
}

// ---------------------------------------------------------------------------
// Top-level VisionEncoder
// ---------------------------------------------------------------------------

/// Gemma4 SigLIP2 vision encoder.
pub struct VisionEncoder {
    patch_embedder: PatchEmbedder,
    layers: Vec<VisionEncoderLayer>,
    embed_proj: Linear,
    pooler: VisionPooler,
    output_length: usize,
    pooling_kernel: usize,
    /// The dtype that all encoder weights were loaded in (e.g. BF16 for GGUF models).
    /// `pixel_values` (always F32) are cast to this dtype before the first matmul.
    weight_dtype: DType,
}

impl VisionEncoder {
    /// Load the vision encoder from a `VarBuilder` rooted at the model prefix.
    ///
    /// Expected call site: `VisionEncoder::load(vb.pp("model"), cfg, lm_hidden_size, device, dtype)`
    pub fn load(
        vb: VarBuilder,
        cfg: &VisionConfig,
        lm_hidden_size: usize,
        _device: &Device,
        dtype: DType,
    ) -> Result<Self> {
        let tower_vb = vb.pp("vision_tower");
        let embedder_vb = tower_vb.pp("patch_embedder");
        let encoder_vb = tower_vb.pp("encoder");

        let patch_embedder = PatchEmbedder::load(embedder_vb, cfg)
            .context("Failed to load vision patch embedder")?;

        let layers: Vec<VisionEncoderLayer> = (0..cfg.num_hidden_layers)
            .map(|i| {
                VisionEncoderLayer::load(encoder_vb.pp(format!("layers.{i}")), cfg)
                    .with_context(|| format!("Failed to load vision encoder layer {i}"))
            })
            .collect::<Result<_>>()?;

        // Projection into LM space: [lm_hidden_size, hidden_size] stored as transposed.
        let embed_proj = linear_no_bias(
            cfg.hidden_size,
            lm_hidden_size,
            vb.pp("embed_vision").pp("embedding_projection"),
        )
        .context("Failed to load embed_vision.embedding_projection")?;

        let pooler = VisionPooler::new(cfg);

        Ok(Self {
            patch_embedder,
            layers,
            embed_proj,
            pooler,
            output_length: cfg.default_output_length,
            pooling_kernel: cfg.pooling_kernel_size,
            weight_dtype: dtype,
        })
    }

    /// Encode a batch of pre-patchified images.
    ///
    /// `pixel_values`:  `[N_patches, patch_size² × 3]` f32, values in [0, 1].
    /// `position_ids`:  `[N_patches, 2]` i64, (x, y) patch grid coordinates.
    ///                  Padding patches have coordinate -1.
    /// `n_soft_tokens`: number of output soft tokens (defaults to `default_output_length`).
    ///
    /// Returns `[n_soft_tokens, lm_hidden_size]`.
    pub fn encode(
        &self,
        pixel_values: &Tensor,
        position_ids: &Tensor,
        n_soft_tokens: Option<usize>,
    ) -> Result<Tensor> {
        let _n_patches = pixel_values.dim(0)?;
        let n_out = n_soft_tokens.unwrap_or(self.output_length);

        // Cast pixel_values to the encoder's weight dtype so that the first
        // matmul (patch projection) does not hit a dtype mismatch.
        // pixel_values arrives as F32 from the server; weights are BF16 when the
        // model was loaded from a GGUF file.
        let pixel_values = if pixel_values.dtype() != self.weight_dtype {
            pixel_values.to_dtype(self.weight_dtype)?
        } else {
            pixel_values.clone()
        };

        // Build padding mask: positions == -1 → true (padding).
        // Sum x+y: if both are -1, sum == -2 (use x alone is simpler).
        let pos_x = position_ids.narrow(1, 0, 1)?.squeeze(1)?; // [N] i64
        let padding_mask = pos_x.lt(0i64)?.to_dtype(pixel_values.dtype())?; // [N] float

        // Embed patches.
        let mut hidden = self
            .patch_embedder
            .forward(&pixel_values, position_ids, &padding_mask)?;

        // Transformer layers.
        for layer in &self.layers {
            hidden = layer.forward(&hidden, position_ids)?;
        }

        // Pool to n_out soft tokens.
        hidden = self.pooler.forward(
            &hidden,
            position_ids,
            &padding_mask,
            n_out,
            self.pooling_kernel,
        )?;

        // Project to LM hidden size.
        self.embed_proj.forward(&hidden).map_err(Into::into)
    }
}

// ---------------------------------------------------------------------------
// RoPE helpers
// ---------------------------------------------------------------------------

/// Build 1/theta^(2i/dim) inverse frequencies on CPU.
fn build_inv_freq(n_freq: usize, theta: f64, device: &Device) -> Result<Tensor> {
    let freqs: Vec<f32> = (0..n_freq)
        .map(|i| {
            let exp = (2 * i) as f64 / (n_freq * 2) as f64;
            (theta.powf(-exp)) as f32
        })
        .collect();
    Ok(Tensor::from_vec(freqs, n_freq, device)?)
}

/// Apply 2D factored RoPE to a `[N, heads, head_dim]` tensor.
///
/// The head_dim is split into two halves: the first half gets x-dimension
/// rotation, the second half gets y-dimension rotation.  Within each half,
/// the standard `[cos, -sin; sin, cos]` rotation is applied using
/// `inv_freq` of shape `[head_dim/4]`.
///
/// All `narrow`/`squeeze`/`broadcast_as` results are made contiguous before
/// use: Metal element-wise and matmul kernels require contiguous inputs.
fn apply_2d_rope(x: &Tensor, position_ids: &Tensor, inv_freq: &Tensor) -> Result<Tensor> {
    let (n, heads, hd) = x.dims3()?;
    let half = hd / 2;

    // Split into x-rot and y-rot halves — contiguous copies for Metal.
    let x_half = x.narrow(D::Minus1, 0, half)?.contiguous()?; // [N, heads, half]
    let y_half = x.narrow(D::Minus1, half, half)?.contiguous()?; // [N, heads, half]

    // Compute cos/sin from position_ids using inv_freq.
    // narrow+squeeze → contiguous before to_dtype (avoids non-contiguous cast on Metal).
    let pos_x = position_ids
        .narrow(1, 0, 1)?
        .squeeze(1)?
        .contiguous()?
        .to_dtype(DType::F32)?; // [N]
    let pos_y = position_ids
        .narrow(1, 1, 1)?
        .squeeze(1)?
        .contiguous()?
        .to_dtype(DType::F32)?; // [N]

    let inv_freq = inv_freq.to_device(x.device())?.to_dtype(DType::F32)?;

    // freqs: [N, n_freq] = outer(pos, inv_freq)
    let freqs_x = pos_x.unsqueeze(1)?.broadcast_mul(&inv_freq.unsqueeze(0)?)?; // [N, n_freq]
    let freqs_y = pos_y.unsqueeze(1)?.broadcast_mul(&inv_freq.unsqueeze(0)?)?; // [N, n_freq]

    // emb = cat([freqs, freqs], dim=-1) → [N, half]  (cat always returns contiguous)
    let emb_x = Tensor::cat(&[&freqs_x, &freqs_x], D::Minus1)?; // [N, half]
    let emb_y = Tensor::cat(&[&freqs_y, &freqs_y], D::Minus1)?; // [N, half]

    let cos_x = emb_x.cos()?.to_dtype(x.dtype())?; // [N, half]
    let sin_x = emb_x.sin()?.to_dtype(x.dtype())?;
    let cos_y = emb_y.cos()?.to_dtype(x.dtype())?;
    let sin_y = emb_y.sin()?.to_dtype(x.dtype())?;

    // broadcast_as produces a non-contiguous (strided) view on Metal;
    // expand+contiguous materialises into a real [N, heads, half] buffer.
    let cos_x = cos_x.unsqueeze(1)?.expand((n, heads, half))?.contiguous()?;
    let sin_x = sin_x.unsqueeze(1)?.expand((n, heads, half))?.contiguous()?;
    let cos_y = cos_y.unsqueeze(1)?.expand((n, heads, half))?.contiguous()?;
    let sin_y = sin_y.unsqueeze(1)?.expand((n, heads, half))?.contiguous()?;

    let x_rot = rotate_half_apply(&x_half, &cos_x, &sin_x)?;
    let y_rot = rotate_half_apply(&y_half, &cos_y, &sin_y)?;

    // Concatenate back.
    Tensor::cat(&[&x_rot, &y_rot], D::Minus1).map_err(Into::into)
}

/// x * cos + rotate_half(x) * sin
///
/// All inputs are expected to be contiguous (caller's responsibility).
/// Internal narrow slices are made contiguous before use.
fn rotate_half_apply(x: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
    let half = x.dim(D::Minus1)? / 2;
    let x1 = x.narrow(D::Minus1, 0, half)?.contiguous()?;
    let x2 = x.narrow(D::Minus1, half, half)?.contiguous()?;
    let rot = Tensor::cat(&[&(x2.neg()?), &x1], D::Minus1)?;
    Ok(((x * cos)? + (rot * sin)?)?)
}

/// Repeat KV heads to match Q heads (GQA).
fn repeat_kv(kv: Tensor, n_rep: usize) -> Result<Tensor> {
    if n_rep == 1 {
        return Ok(kv);
    }
    let (heads, n, hd) = kv.dims3()?;
    kv.unsqueeze(1)?
        .expand((heads, n_rep, n, hd))?
        .reshape((heads * n_rep, n, hd))
        .map_err(Into::into)
}
