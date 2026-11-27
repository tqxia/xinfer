//! Protocol-agnostic request, sampling, streaming event, and model metadata types.

use serde::{Deserialize, Serialize};
use xinfer_runtime::{require_kind, Error, ErrorKind, ModelType, Result};

/// Per-request sampling parameters. `temperature == 0.0` selects deterministic greedy argmax.
#[derive(Clone, Debug)]
pub struct SamplingParams {
    pub temperature: f32,
    pub top_p: f32,
    pub top_k: i32,
    pub seed: Option<u64>,
}

impl Default for SamplingParams {
    fn default() -> Self {
        Self {
            temperature: 0.0,
            top_p: 1.0,
            top_k: 0,
            seed: None,
        }
    }
}

impl SamplingParams {
    pub fn validate(&self) -> Result<()> {
        require_kind(
            self.temperature.is_finite() && self.temperature >= 0.0,
            ErrorKind::InvalidInput,
            "temperature must be finite and >= 0.0",
        )?;
        require_kind(
            self.top_p.is_finite() && self.top_p > 0.0 && self.top_p <= 1.0,
            ErrorKind::InvalidInput,
            "top_p must be finite and in (0.0, 1.0]",
        )?;
        require_kind(
            self.top_k >= 0,
            ErrorKind::InvalidInput,
            "top_k must be >= 0",
        )?;
        Ok(())
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
}

#[derive(Clone, Debug)]
pub struct GenerateRequest {
    pub prompt_tokens: Vec<i64>,
    pub max_tokens: usize,
    pub sampling: SamplingParams,
    pub stop_strings: Vec<String>,
    pub stop_token_ids: Vec<i64>,
    pub ignore_eos: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FinishReason {
    Stop,
    Length,
}

impl FinishReason {
    pub fn as_openai_str(self) -> &'static str {
        match self {
            Self::Stop => "stop",
            Self::Length => "length",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Usage {
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
}

impl Usage {
    pub fn total_tokens(self) -> usize {
        self.prompt_tokens + self.completion_tokens
    }
}

#[derive(Debug)]
pub enum GenerateEvent {
    Delta { token_id: i64, text: String },
    Finished { reason: FinishReason, usage: Usage },
    Error(Error),
}

#[derive(Clone, Debug)]
pub struct ModelInfo {
    pub model_id: String,
    pub model_type: ModelType,
    pub cache_length: usize,
    pub max_prefill_tokens: usize,
    pub allocated_batch_size: usize,
    pub vocab_size: usize,
}
