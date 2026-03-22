//! Continuous batching scheduler with chunked prefill.
//!
//! The scheduler manages sequences across prefill and decode phases,
//! deciding which sequences get tokens in each step.

use std::collections::VecDeque;

use crate::sampler::SamplingParams;

/// Status of a sequence in the scheduler.
#[derive(Debug, Clone, PartialEq)]
#[allow(dead_code)]
pub enum SequenceStatus {
    /// Waiting to be scheduled for prefill.
    Waiting,
    /// Currently running (prefilling or decoding).
    Running,
    /// Finished generation.
    Finished(FinishReason),
}

/// Why a sequence finished.
#[derive(Debug, Clone, PartialEq)]
#[allow(dead_code)]
pub enum FinishReason {
    Stop,
    Length,
    Abort,
}

impl FinishReason {
    #[allow(dead_code)]
    pub fn as_str(&self) -> &'static str {
        match self {
            FinishReason::Stop => "stop",
            FinishReason::Length => "length",
            FinishReason::Abort => "abort",
        }
    }
}

/// A sequence being tracked by the scheduler.
#[derive(Debug)]
#[allow(dead_code)]
pub struct Sequence {
    pub id: String,
    pub prompt_token_ids: Vec<u32>,
    pub output_token_ids: Vec<u32>,
    pub status: SequenceStatus,
    pub sampling_params: SamplingParams,
    /// How many prompt tokens have been processed so far.
    pub num_computed_tokens: usize,
    /// EOS token ID for this model.
    pub eos_token_id: Option<u32>,
}

impl Sequence {
    #[allow(dead_code)]
    pub fn new(
        id: String,
        prompt_token_ids: Vec<u32>,
        sampling_params: SamplingParams,
        eos_token_id: Option<u32>,
    ) -> Self {
        Self {
            id,
            prompt_token_ids,
            output_token_ids: Vec::new(),
            status: SequenceStatus::Waiting,
            sampling_params,
            num_computed_tokens: 0,
            eos_token_id,
        }
    }

    /// Total tokens so far (prompt + generated).
    #[allow(dead_code)]
    pub fn total_tokens(&self) -> usize {
        self.prompt_token_ids.len() + self.output_token_ids.len()
    }

    /// All token IDs (prompt + output).
    #[allow(dead_code)]
    pub fn all_token_ids(&self) -> Vec<u32> {
        let mut ids = self.prompt_token_ids.clone();
        ids.extend_from_slice(&self.output_token_ids);
        ids
    }

    /// Check if the sequence should stop.
    #[allow(dead_code)]
    pub fn should_stop(&self) -> bool {
        // Check max tokens
        if self.output_token_ids.len() >= self.sampling_params.max_tokens {
            return true;
        }
        // Check EOS token
        if let Some(eos_id) = self.eos_token_id {
            if let Some(&last) = self.output_token_ids.last() {
                if last == eos_id {
                    return true;
                }
            }
        }
        false
    }

    /// Determine the finish reason.
    #[allow(dead_code)]
    pub fn finish_reason(&self) -> FinishReason {
        if let Some(eos_id) = self.eos_token_id {
            if let Some(&last) = self.output_token_ids.last() {
                if last == eos_id {
                    return FinishReason::Stop;
                }
            }
        }
        FinishReason::Length
    }
}

/// What the scheduler decided for one step.
#[allow(dead_code)]
pub struct SchedulerOutput {
    /// Sequences to process in this step.
    pub sequences: Vec<ScheduledSequence>,
}

/// A sequence scheduled for this step.
#[allow(dead_code)]
pub struct ScheduledSequence {
    /// Index into the scheduler's sequence list.
    pub seq_idx: usize,
    /// Whether this is a prefill step (vs decode).
    pub is_prefill: bool,
    /// Number of tokens to process in this step.
    pub num_tokens: usize,
}

/// The scheduler.
#[allow(dead_code)]
pub struct Scheduler {
    /// All sequences.
    pub sequences: Vec<Sequence>,
    /// Indices of waiting sequences (FIFO).
    waiting: VecDeque<usize>,
    /// Indices of running sequences.
    running: Vec<usize>,
    /// Max concurrent sequences.
    max_batch_size: usize,
    /// Max tokens per step.
    max_tokens_per_step: usize,
}

impl Scheduler {
    #[allow(dead_code)]
    pub fn new(max_batch_size: usize, max_tokens_per_step: usize) -> Self {
        Self {
            sequences: Vec::new(),
            waiting: VecDeque::new(),
            running: Vec::new(),
            max_batch_size,
            max_tokens_per_step,
        }
    }

    /// Add a new request.
    #[allow(dead_code)]
    pub fn add_request(&mut self, seq: Sequence) -> usize {
        let idx = self.sequences.len();
        self.sequences.push(seq);
        self.waiting.push_back(idx);
        idx
    }

    /// Schedule the next step.
    ///
    /// Returns which sequences to process and how many tokens each gets.
    #[allow(dead_code)]
    pub fn schedule(&mut self) -> SchedulerOutput {
        let mut scheduled = Vec::new();
        let mut token_budget = self.max_tokens_per_step;

        // First: schedule running sequences (decode step = 1 token each)
        let mut still_running = Vec::new();
        for &idx in &self.running {
            if token_budget == 0 {
                still_running.push(idx);
                continue;
            }
            let seq = &self.sequences[idx];
            if matches!(seq.status, SequenceStatus::Running) {
                scheduled.push(ScheduledSequence {
                    seq_idx: idx,
                    is_prefill: false,
                    num_tokens: 1,
                });
                token_budget = token_budget.saturating_sub(1);
                still_running.push(idx);
            }
        }
        self.running = still_running;

        // Second: schedule waiting sequences (prefill)
        while !self.waiting.is_empty()
            && self.running.len() < self.max_batch_size
            && token_budget > 0
        {
            let idx = self.waiting.pop_front().unwrap();
            let seq = &mut self.sequences[idx];

            let remaining_prompt = seq
                .prompt_token_ids
                .len()
                .saturating_sub(seq.num_computed_tokens);
            let num_tokens = remaining_prompt.min(token_budget);

            if num_tokens > 0 {
                seq.status = SequenceStatus::Running;
                scheduled.push(ScheduledSequence {
                    seq_idx: idx,
                    is_prefill: true,
                    num_tokens,
                });
                token_budget = token_budget.saturating_sub(num_tokens);
                self.running.push(idx);
            }
        }

        SchedulerOutput {
            sequences: scheduled,
        }
    }

    /// Update scheduler state after a step.
    /// Called with the generated token for each sequence.
    #[allow(dead_code)]
    pub fn update_after_step(&mut self, results: &[(usize, u32)]) {
        let mut finished_indices = Vec::new();

        for &(idx, token_id) in results {
            let seq = &mut self.sequences[idx];
            seq.output_token_ids.push(token_id);
            seq.num_computed_tokens = seq.prompt_token_ids.len() + seq.output_token_ids.len();

            if seq.should_stop() {
                let reason = seq.finish_reason();
                seq.status = SequenceStatus::Finished(reason);
                finished_indices.push(idx);
            }
        }

        // Remove finished sequences from running list
        self.running.retain(|idx| !finished_indices.contains(idx));
    }

    /// Check if there's any work to do.
    #[allow(dead_code)]
    pub fn has_work(&self) -> bool {
        !self.waiting.is_empty() || !self.running.is_empty()
    }

    /// Get a sequence by index.
    #[allow(dead_code)]
    pub fn get_sequence(&self, idx: usize) -> &Sequence {
        &self.sequences[idx]
    }

    /// Check if a sequence is finished.
    #[allow(dead_code)]
    pub fn is_finished(&self, idx: usize) -> bool {
        matches!(self.sequences[idx].status, SequenceStatus::Finished(_))
    }
}
