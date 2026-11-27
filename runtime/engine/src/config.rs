//! Configuration for the continuous-batching inference engine.

use xinfer_runtime::{require_kind, Error, ErrorKind, Result, Spec};

#[derive(Clone, Debug)]
pub struct EngineConfig {
    /// Number of batch slots to allocate (`1..=spec.max_batch_size`). Defaults to `spec.max_batch_size`.
    pub allocated_batch_size: Option<usize>,
    /// Maximum prompt tokens processed per interleaved prefill chunk (`2..=spec.max_prefill`).
    pub prefill_chunk_size: Option<usize>,
    /// Number of batched decode steps executed between interleaved prefill chunks when both exist.
    pub decode_steps_per_prefill_chunk: usize,
    /// Maximum waiting requests buffered before rejecting new arrivals.
    pub max_queued_requests: usize,
    /// Optional model ID override advertised by `ModelInfo` / `/v1/models`.
    pub served_model_name: Option<String>,
}

impl Default for EngineConfig {
    fn default() -> Self {
        Self {
            allocated_batch_size: None,
            prefill_chunk_size: None,
            decode_steps_per_prefill_chunk: 1,
            max_queued_requests: 64,
            served_model_name: None,
        }
    }
}

/// Validated, concrete engine parameters resolved against a bundle `Spec`.
#[derive(Clone, Debug)]
pub struct ResolvedEngineConfig {
    pub allocated_batch_size: usize,
    pub prefill_chunk_size: usize,
    pub decode_steps_per_prefill_chunk: usize,
    pub max_queued_requests: usize,
    pub served_model_name: String,
}

impl EngineConfig {
    pub fn resolve(&self, spec: &Spec) -> Result<ResolvedEngineConfig> {
        let max_batch = spec.max_batch_size as usize;
        let allocated_batch_size = self.allocated_batch_size.unwrap_or(max_batch);
        require_kind(
            allocated_batch_size >= 1 && allocated_batch_size <= max_batch,
            ErrorKind::InvalidInput,
            format!("allocated_batch_size must be in 1..={max_batch}"),
        )?;

        let max_prefill = spec.max_prefill as usize;
        let prefill_chunk_size = self.prefill_chunk_size.unwrap_or(max_prefill);
        require_kind(
            prefill_chunk_size >= 2 && prefill_chunk_size <= max_prefill,
            ErrorKind::InvalidInput,
            format!("prefill_chunk_size must be in 2..={max_prefill}"),
        )?;

        require_kind(
            self.decode_steps_per_prefill_chunk >= 1,
            ErrorKind::InvalidInput,
            "decode_steps_per_prefill_chunk must be >= 1",
        )?;
        require_kind(
            self.max_queued_requests >= 1,
            ErrorKind::InvalidInput,
            "max_queued_requests must be >= 1",
        )?;

        let served_model_name = match &self.served_model_name {
            Some(name) if !name.trim().is_empty() => name.clone(),
            Some(_) => return Err(Error::invalid_input("served_model_name must not be empty")),
            None => {
                let model = spec.checkpoint_model();
                if model.is_empty() {
                    spec.model_type.as_str().to_string()
                } else {
                    model.to_string()
                }
            }
        };

        Ok(ResolvedEngineConfig {
            allocated_batch_size,
            prefill_chunk_size,
            decode_steps_per_prefill_chunk: self.decode_steps_per_prefill_chunk,
            max_queued_requests: self.max_queued_requests,
            served_model_name,
        })
    }
}
