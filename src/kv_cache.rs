//! Block-based KV cache with grow-on-demand allocation.
//!
//! This module is a placeholder for the full block-based KV cache system.
//! Currently, the candle-transformers models manage their own concat-based
//! KV caches internally. This module tracks memory usage and provides the
//! interface for future paged-attention integration.

/// KV cache configuration.
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct KvCacheConfig {
    pub block_size: usize,
    pub initial_blocks: usize,
    pub max_blocks: usize, // 0 = no limit
}

impl KvCacheConfig {
    #[allow(dead_code)]
    pub fn new(block_size: usize, initial_blocks: usize, max_blocks: usize) -> Self {
        Self {
            block_size,
            initial_blocks,
            max_blocks,
        }
    }
}

/// Tracks KV cache usage for a single sequence.
#[derive(Debug)]
#[allow(dead_code)]
pub struct SequenceCache {
    /// Number of tokens currently cached.
    pub num_cached_tokens: usize,
    /// Maximum tokens this cache can hold before needing more blocks.
    pub max_tokens: usize,
}

impl SequenceCache {
    #[allow(dead_code)]
    pub fn new(config: &KvCacheConfig) -> Self {
        Self {
            num_cached_tokens: 0,
            max_tokens: config.initial_blocks * config.block_size,
        }
    }

    /// Check if we can cache more tokens.
    #[allow(dead_code)]
    pub fn can_grow(&self, config: &KvCacheConfig) -> bool {
        if config.max_blocks == 0 {
            return true; // No limit
        }
        let max_possible = config.max_blocks * config.block_size;
        self.max_tokens < max_possible
    }

    /// Record that we cached additional tokens.
    #[allow(dead_code)]
    pub fn record_tokens(&mut self, count: usize) {
        self.num_cached_tokens += count;
    }
}
