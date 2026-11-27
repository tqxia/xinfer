//! Protocol-agnostic continuous-batching inference engine for `xinfer`.
//!
//! Owns the dedicated GPU actor thread (`Session` is `!Send + !Sync`), slot table
//! with interleaved chunked prefill, GPU-assisted sampling, and bundled native
//! Rust tokenization / Jinja chat templating.

pub mod actor;
pub mod config;
pub mod sampler;
pub mod scheduler;
pub mod tokenizer;
pub mod types;

pub use actor::{Engine, EngineHandle};
pub use config::{EngineConfig, ResolvedEngineConfig};
pub use sampler::{sample_batch, RequestRng, SampleRow};
pub use scheduler::{BatchBackend, PendingRequest, Scheduler, SessionBackend};
pub use tokenizer::{BundleTokenizer, StreamDecoder, StreamStep};
pub use types::{
    ChatMessage, FinishReason, GenerateEvent, GenerateRequest, ModelInfo, SamplingParams, Usage,
};

#[cfg(test)]
mod tests;
