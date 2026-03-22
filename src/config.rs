//! Model configuration loading from config.json.

use anyhow::{Context, Result};
use serde::Deserialize;
use std::path::Path;

/// Supported model architectures.
#[derive(Debug, Clone, PartialEq)]
pub enum ModelArchitecture {
    Qwen2,
    Gemma2,
    Gemma3,
}

/// Raw config.json from HuggingFace.
#[derive(Debug, Deserialize)]
pub struct RawConfig {
    pub architectures: Option<Vec<String>>,
    pub model_type: Option<String>,
    pub vocab_size: Option<usize>,
    pub hidden_size: Option<usize>,
    pub intermediate_size: Option<usize>,
    pub num_hidden_layers: Option<usize>,
    pub num_attention_heads: Option<usize>,
    pub num_key_value_heads: Option<usize>,
    pub max_position_embeddings: Option<usize>,
    pub rms_norm_eps: Option<f64>,
    pub rope_theta: Option<f64>,
    pub tie_word_embeddings: Option<bool>,
    #[allow(dead_code)]
    pub hidden_act: Option<String>,

    // Qwen2-specific
    pub sliding_window: Option<usize>,
    pub max_window_layers: Option<usize>,
    pub use_sliding_window: Option<bool>,

    // Gemma2-specific
    pub head_dim: Option<usize>,
    pub hidden_activation: Option<String>,
    pub attention_bias: Option<bool>,
    pub final_logit_softcapping: Option<f64>,
    pub attn_logit_softcapping: Option<f64>,
    pub query_pre_attn_scalar: Option<usize>,

    // Gemma3-specific
    pub sliding_window_pattern: Option<usize>,
}

impl RawConfig {
    pub fn from_file(path: &Path) -> Result<Self> {
        let content = std::fs::read_to_string(path).context("Failed to read config.json")?;
        let config: RawConfig =
            serde_json::from_str(&content).context("Failed to parse config.json")?;
        Ok(config)
    }

    pub fn detect_architecture(&self) -> Result<ModelArchitecture> {
        if let Some(archs) = &self.architectures {
            for arch in archs {
                if arch.contains("Qwen2") || arch.contains("Qwen3") {
                    return Ok(ModelArchitecture::Qwen2);
                }
                if arch.contains("Gemma3") {
                    return Ok(ModelArchitecture::Gemma3);
                }
                if arch.contains("Gemma2") {
                    return Ok(ModelArchitecture::Gemma2);
                }
            }
        }

        if let Some(model_type) = &self.model_type {
            match model_type.as_str() {
                "qwen2" | "qwen2_5" | "qwen3" | "qwen3_5" => return Ok(ModelArchitecture::Qwen2),
                "gemma3" => return Ok(ModelArchitecture::Gemma3),
                "gemma2" => return Ok(ModelArchitecture::Gemma2),
                _ => {}
            }
        }

        anyhow::bail!(
            "Unsupported model architecture. architectures={:?}, model_type={:?}",
            self.architectures,
            self.model_type
        )
    }

    pub fn to_qwen2_config(&self) -> candle_transformers::models::qwen2::Config {
        candle_transformers::models::qwen2::Config {
            vocab_size: self.vocab_size.unwrap_or(151936),
            hidden_size: self.hidden_size.unwrap_or(896),
            intermediate_size: self.intermediate_size.unwrap_or(4864),
            num_hidden_layers: self.num_hidden_layers.unwrap_or(24),
            num_attention_heads: self.num_attention_heads.unwrap_or(14),
            num_key_value_heads: self.num_key_value_heads.unwrap_or(2),
            max_position_embeddings: self.max_position_embeddings.unwrap_or(131072),
            sliding_window: self.sliding_window.unwrap_or(131072),
            max_window_layers: self.max_window_layers.unwrap_or(28),
            tie_word_embeddings: self.tie_word_embeddings.unwrap_or(true),
            rope_theta: self.rope_theta.unwrap_or(1000000.0),
            rms_norm_eps: self.rms_norm_eps.unwrap_or(1e-6),
            use_sliding_window: self.use_sliding_window.unwrap_or(false),
            hidden_act: candle_nn::Activation::Silu,
        }
    }

    pub fn to_gemma2_config(&self) -> candle_transformers::models::gemma2::Config {
        let num_attention_heads = self.num_attention_heads.unwrap_or(16);
        let hidden_size = self.hidden_size.unwrap_or(3584);
        candle_transformers::models::gemma2::Config {
            vocab_size: self.vocab_size.unwrap_or(256000),
            hidden_size,
            intermediate_size: self.intermediate_size.unwrap_or(14336),
            num_hidden_layers: self.num_hidden_layers.unwrap_or(42),
            num_attention_heads,
            num_key_value_heads: self.num_key_value_heads.unwrap_or(8),
            head_dim: self.head_dim.unwrap_or(hidden_size / num_attention_heads),
            hidden_activation: parse_gemma_activation(
                self.hidden_activation
                    .as_deref()
                    .unwrap_or("gelu_pytorch_tanh"),
            ),
            rms_norm_eps: self.rms_norm_eps.unwrap_or(1e-6),
            rope_theta: self.rope_theta.unwrap_or(10000.0),
            attention_bias: self.attention_bias.unwrap_or(false),
            final_logit_softcapping: self.final_logit_softcapping,
            attn_logit_softcapping: self.attn_logit_softcapping,
            query_pre_attn_scalar: self.query_pre_attn_scalar.unwrap_or(256),
            sliding_window: self.sliding_window,
            max_position_embeddings: self.max_position_embeddings.unwrap_or(8192),
        }
    }

    pub fn to_gemma3_config(&self) -> candle_transformers::models::gemma3::Config {
        let num_attention_heads = self.num_attention_heads.unwrap_or(8);
        let hidden_size = self.hidden_size.unwrap_or(2560);
        candle_transformers::models::gemma3::Config {
            vocab_size: self.vocab_size.unwrap_or(262144),
            hidden_size,
            intermediate_size: self.intermediate_size.unwrap_or(10240),
            num_hidden_layers: self.num_hidden_layers.unwrap_or(34),
            num_attention_heads,
            num_key_value_heads: self.num_key_value_heads.unwrap_or(4),
            head_dim: self.head_dim.unwrap_or(256),
            hidden_activation: parse_gemma_activation(
                self.hidden_activation
                    .as_deref()
                    .unwrap_or("gelu_pytorch_tanh"),
            ),
            rms_norm_eps: self.rms_norm_eps.unwrap_or(1e-6),
            rope_theta: self.rope_theta.unwrap_or(10000.0),
            attention_bias: self.attention_bias.unwrap_or(false),
            final_logit_softcapping: self.final_logit_softcapping,
            attn_logit_softcapping: self.attn_logit_softcapping,
            query_pre_attn_scalar: self.query_pre_attn_scalar.unwrap_or(256),
            sliding_window: self.sliding_window.unwrap_or(1024),
            sliding_window_pattern: self.sliding_window_pattern.unwrap_or(6),
            max_position_embeddings: self.max_position_embeddings.unwrap_or(131072),
        }
    }

    #[allow(dead_code)]
    pub fn max_seq_len(&self) -> usize {
        self.max_position_embeddings.unwrap_or(4096)
    }
}

fn parse_gemma_activation(s: &str) -> candle_nn::Activation {
    match s {
        "gelu_pytorch_tanh" | "gelu_fast" | "gelu_new" => candle_nn::Activation::GeluPytorchTanh,
        "gelu" => candle_nn::Activation::Gelu,
        "silu" => candle_nn::Activation::Silu,
        "relu" => candle_nn::Activation::Relu,
        _ => candle_nn::Activation::GeluPytorchTanh,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn config_with_arch(archs: &[&str], model_type: &str) -> RawConfig {
        RawConfig {
            architectures: Some(archs.iter().map(|s| s.to_string()).collect()),
            model_type: Some(model_type.to_string()),
            vocab_size: None,
            hidden_size: None,
            intermediate_size: None,
            num_hidden_layers: None,
            num_attention_heads: None,
            num_key_value_heads: None,
            max_position_embeddings: None,
            rms_norm_eps: None,
            rope_theta: None,
            tie_word_embeddings: None,
            hidden_act: None,
            sliding_window: None,
            max_window_layers: None,
            use_sliding_window: None,
            head_dim: None,
            hidden_activation: None,
            attention_bias: None,
            final_logit_softcapping: None,
            attn_logit_softcapping: None,
            query_pre_attn_scalar: None,
            sliding_window_pattern: None,
        }
    }

    #[test]
    fn detect_qwen2_architecture() {
        let cfg = config_with_arch(&["Qwen2ForCausalLM"], "qwen2");
        assert_eq!(cfg.detect_architecture().unwrap(), ModelArchitecture::Qwen2);
    }

    #[test]
    fn detect_gemma2_architecture() {
        let cfg = config_with_arch(&["Gemma2ForCausalLM"], "gemma2");
        assert_eq!(
            cfg.detect_architecture().unwrap(),
            ModelArchitecture::Gemma2
        );
    }

    #[test]
    fn detect_gemma3_architecture() {
        let cfg = config_with_arch(&["Gemma3ForCausalLM"], "gemma3");
        assert_eq!(
            cfg.detect_architecture().unwrap(),
            ModelArchitecture::Gemma3
        );
    }

    #[test]
    fn detect_gemma3_preferred_over_gemma2() {
        // A config with both Gemma3 and Gemma2 in the architecture list should pick Gemma3.
        let cfg = config_with_arch(&["Gemma3ForCausalLM", "Gemma2ForCausalLM"], "gemma3");
        assert_eq!(
            cfg.detect_architecture().unwrap(),
            ModelArchitecture::Gemma3
        );
    }

    #[test]
    fn unsupported_architecture_errors() {
        let cfg = config_with_arch(&["LlamaForCausalLM"], "llama");
        assert!(cfg.detect_architecture().is_err());
    }

    #[test]
    fn gemma3_config_defaults_are_reasonable() {
        let cfg = config_with_arch(&["Gemma3ForCausalLM"], "gemma3");
        let g3 = cfg.to_gemma3_config();
        assert!(g3.num_hidden_layers > 0);
        assert!(g3.hidden_size > 0);
        assert!(g3.sliding_window > 0);
        assert!(g3.sliding_window_pattern > 0);
    }
}
