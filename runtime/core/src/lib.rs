//! Rust-core inference runtime for xinfer AOTI model bundles.
//!
//! Drives compiled `prefill.pt2` / `decode.pt2` packages and shared
//! `weights.safetensors` directly over PyTorch 2.13's `aoti_torch_*` C ABI
//! and `libcudart`, with a minimal static C++ shim for `AOTIModelPackageLoader`.

pub mod error;
pub mod io;
pub mod runner;
pub mod sha256;
pub mod spec;
pub mod state;
pub mod tensor;
pub mod torch_sys;
pub mod weights;

pub use error::{require, require_kind, Error, ErrorKind, Result};
pub use io::{
    create_output_file, path_exists, weakly_canonical, write_output_file, write_tensor_bytes,
};
pub use runner::{
    generate_with, prefill_with, write_logits_f32, Model, Package, Runner, Session, StepDriver,
};
pub use sha256::sha256_file;
pub use spec::{
    checked_device, configured_device, metadata_check, metadata_digest, metadata_integer,
    metadata_keys, parse_integer, read_json, torch_build_info, validate_metadata, ModelType, Spec,
    StateSpec, TorchBuildInfo, MAX_BATCH_SIZE, MAX_CACHE_LENGTH, MAX_PREFILL_TOKENS,
    METADATA_FIELDS, METADATA_FILE, TOKENIZER_OPTIONAL_FILES, TOKENIZER_REQUIRED_FILES,
};
pub use state::StatePolicy;
pub use tensor::{
    contiguous_strides, cuda_device_capability, cuda_mem_info, cuda_synchronize,
    current_cuda_stream, DType, Device, Tensor,
};
pub use weights::{
    check_view_bounds, read_safetensors_metadata, SharedWeights, ValidatedBundle,
    SHARED_WEIGHTS_FILE, SHARED_WEIGHTS_MANIFEST_KEY,
};

#[cfg(test)]
mod tests;
