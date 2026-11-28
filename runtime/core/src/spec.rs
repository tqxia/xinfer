//! Package `metadata.json` validation, model-specific state geometry checks,
//! and request / token-ID bounds enforcement (all CPU-only until `checked_device`).

use std::collections::HashSet;
use std::ffi::CStr;
use std::fs;
use std::path::Path;

use serde_json::Value;

use crate::error::{require, require_kind, Error, ErrorKind, Result};
use crate::tensor::{contiguous_strides, cuda_device_capability, DType, Device};
use crate::torch_sys::{self, shim_call, XinferTorchInfo};
use crate::weights::ValidatedBundle;

pub const METADATA_FILE: &str = "metadata.json";
pub const MAX_BATCH_SIZE: i64 = 8;
pub const MAX_PREFILL_TOKENS: i64 = 512;
// Keep cumulative int32 KV lengths across all compiled batch slots below i32::MAX.
pub const MAX_CACHE_LENGTH: i64 = (i32::MAX as i64) / MAX_BATCH_SIZE;

pub const TOKENIZER_REQUIRED_FILES: [&str; 2] = ["tokenizer.json", "tokenizer_config.json"];
pub const TOKENIZER_OPTIONAL_FILES: [&str; 1] = ["chat_template.jinja"];

pub const METADATA_FIELDS: [&str; 13] = [
    "model_type",
    "torch_version",
    "cxx11_abi",
    "cuda_compute_capability",
    "cache_length",
    "max_prefill_tokens",
    "vocab_size",
    "eos_token_ids",
    "checkpoint",
    "state",
    "entries",
    "shared_weights",
    "tokenizer",
];

/// Logical model family declared in `metadata.json` (`model_type`).
/// `Qwen35` and `Qwen38` share the same hybrid state ABI while retaining distinct identities.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ModelType {
    Qwen3,
    Qwen35,
    Qwen38,
}

impl ModelType {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Qwen3 => "qwen3",
            Self::Qwen35 => "qwen35",
            Self::Qwen38 => "qwen38",
        }
    }

    pub fn parse(value: &str) -> Result<Self> {
        match value {
            "qwen3" => Ok(Self::Qwen3),
            "qwen35" => Ok(Self::Qwen35),
            "qwen38" => Ok(Self::Qwen38),
            _ => Err(Error::new("Unsupported model type")),
        }
    }

    pub fn from_identity(identity: &Value) -> Result<Self> {
        let model = identity
            .get("model")
            .and_then(Value::as_str)
            .ok_or_else(|| Error::new("Unsupported model identity"))?;
        match model {
            "qwen3" => Ok(Self::Qwen3),
            "qwen3.5" => Ok(Self::Qwen35),
            "qwen3.8" => Ok(Self::Qwen38),
            _ => Err(Error::new(format!("Unsupported model identity: {model}"))),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TorchBuildInfo {
    pub torch_version: String,
    pub cudart_version: i64,
    pub cxx11_abi: bool,
}

/// Queries the compile-time LibTorch version, CUDART version, and C++11 ABI flag.
pub fn torch_build_info() -> TorchBuildInfo {
    let mut info = XinferTorchInfo {
        torch_version: std::ptr::null(),
        cudart_version: 0,
        cxx11_abi: false,
    };
    // SAFETY: `info` is a valid writable `XinferTorchInfo`; shim writes static pointers.
    unsafe {
        torch_sys::xinfer_shim_torch_info(&mut info);
    }
    let torch_version = if info.torch_version.is_null() {
        String::new()
    } else {
        // SAFETY: `TORCH_VERSION` is a static NUL-terminated C string.
        unsafe { CStr::from_ptr(info.torch_version) }
            .to_string_lossy()
            .into_owned()
    };
    TorchBuildInfo {
        torch_version,
        cudart_version: info.cudart_version as i64,
        cxx11_abi: info.cxx11_abi,
    }
}

/// Parses a signed base-10 `i64` matching C++ `std::from_chars` (rejects leading `+`).
pub fn parse_integer(text: &str) -> Result<i64> {
    if text.is_empty() || text.starts_with('+') {
        return Err(Error::new(format!("Invalid integer: {text}")));
    }
    text.parse::<i64>()
        .map_err(|_| Error::new(format!("Invalid integer: {text}")))
}

pub fn read_json(path: &Path, prefix: &str) -> Result<Value> {
    let content =
        fs::read_to_string(path).map_err(|_| Error::new(format!("{prefix}{}", path.display())))?;
    let value: Value = serde_json::from_str(&content)?;
    Ok(value)
}

pub fn metadata_check(condition: bool, field: &str) -> Result<()> {
    require_kind(
        condition,
        ErrorKind::Unsupported,
        format!("Unsupported package metadata: {field}"),
    )
}

/// True iff `value` is a JSON integer (rejecting floats like `1.0`) `>= minimum`.
pub fn metadata_integer(value: &Value, minimum: i64) -> bool {
    value
        .as_i64()
        .map(|integer| integer >= minimum)
        .unwrap_or(false)
}

pub fn metadata_digest(value: &Value) -> bool {
    let Some(text) = value.as_str() else {
        return false;
    };
    text.len() == 64
        && text
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

pub fn metadata_keys(value: &Value, keys: &[&str]) -> bool {
    let Some(obj) = value.as_object() else {
        return false;
    };
    obj.len() == keys.len() && keys.iter().all(|key| obj.contains_key(*key))
}

/// Enforces the strict 13-field `metadata.json` schema before any file or GPU I/O.
pub fn validate_metadata(metadata: &Value) -> Result<()> {
    let obj = metadata
        .as_object()
        .ok_or_else(|| Error::new("Unsupported package metadata: expected an object"))?;

    let model_type = obj.get("model_type").and_then(Value::as_str);
    require(
        matches!(model_type, Some("qwen3" | "qwen35" | "qwen38")),
        "Unsupported model type",
    )?;

    for field in METADATA_FIELDS {
        metadata_check(obj.contains_key(field), field)?;
    }
    for field in obj.keys() {
        metadata_check(METADATA_FIELDS.contains(&field.as_str()), field)?;
    }

    let cache_length_val = &obj["cache_length"];
    metadata_check(
        metadata_integer(cache_length_val, MAX_PREFILL_TOKENS)
            && cache_length_val.as_i64().unwrap_or(0) <= MAX_CACHE_LENGTH,
        "cache_length",
    )?;
    let cache_length = cache_length_val.as_i64().unwrap();

    let max_prefill_val = &obj["max_prefill_tokens"];
    metadata_check(
        metadata_integer(max_prefill_val, 1)
            && max_prefill_val.as_i64() == Some(MAX_PREFILL_TOKENS)
            && max_prefill_val.as_i64().unwrap_or(0) <= cache_length,
        "max_prefill_tokens",
    )?;

    let vocab_val = &obj["vocab_size"];
    metadata_check(metadata_integer(vocab_val, 1), "vocab_size")?;
    let vocab_size = vocab_val.as_i64().unwrap();

    metadata_check(
        obj["torch_version"]
            .as_str()
            .map(|s| !s.is_empty())
            .unwrap_or(false),
        "torch_version",
    )?;
    metadata_check(obj["cxx11_abi"].is_boolean(), "cxx11_abi")?;

    let sm = obj["cuda_compute_capability"].as_array();
    metadata_check(
        sm.map(|arr| {
            arr.len() == 2 && metadata_integer(&arr[0], 1) && metadata_integer(&arr[1], 0)
        })
        .unwrap_or(false),
        "cuda_compute_capability",
    )?;

    let eos = obj["eos_token_ids"].as_array();
    metadata_check(
        eos.map(|arr| !arr.is_empty()).unwrap_or(false),
        "eos_token_ids",
    )?;
    for token in eos.unwrap() {
        metadata_check(
            metadata_integer(token, 0) && token.as_i64().unwrap_or(vocab_size) < vocab_size,
            "eos_token_ids",
        )?;
    }

    let checkpoint = obj["checkpoint"].as_object();
    metadata_check(checkpoint.is_some(), "checkpoint")?;
    let checkpoint = checkpoint.unwrap();
    for field in ["model", "revision"] {
        metadata_check(
            checkpoint
                .get(field)
                .and_then(Value::as_str)
                .map(|s| !s.is_empty())
                .unwrap_or(false),
            "checkpoint",
        )?;
    }

    let states = obj["state"].as_array();
    metadata_check(states.map(|arr| !arr.is_empty()).unwrap_or(false), "state")?;
    let mut names = HashSet::new();
    for item in states.unwrap() {
        metadata_check(
            metadata_keys(item, &["layer", "kind", "shape", "dtype"]),
            "state",
        )?;
        let kind = item["kind"].as_str();
        metadata_check(
            kind.is_some() && metadata_integer(&item["layer"], 0),
            "state",
        )?;
        let kind = kind.unwrap();
        let layer = item["layer"].as_i64().unwrap();
        let kv = kind == "key" || kind == "value";
        metadata_check(kv || kind == "conv" || kind == "recurrent", "state")?;
        let name = format!("layer_{layer}.{kind}");
        metadata_check(names.insert(name), "state")?;

        let shape = item["shape"].as_array();
        let expected_rank = if kind == "conv" { 3 } else { 4 };
        metadata_check(
            shape.map(|arr| arr.len() == expected_rank).unwrap_or(false),
            "state",
        )?;
        let shape = shape.unwrap();
        let mut elements: i64 = 1;
        for dim in shape {
            metadata_check(
                metadata_integer(dim, 1)
                    && dim.as_i64().unwrap_or(i64::MAX) <= (i64::MAX / 4) / elements,
                "state",
            )?;
            elements *= dim.as_i64().unwrap();
        }
        metadata_check(shape[0].as_i64() == Some(MAX_BATCH_SIZE), "state")?;
        let expected_dtype = if kind == "recurrent" {
            "float32"
        } else {
            "bfloat16"
        };
        metadata_check(item["dtype"].as_str() == Some(expected_dtype), "state")?;
        if kv {
            metadata_check(shape[2].as_i64() == Some(cache_length), "state")?;
        }
    }

    let entries = &obj["entries"];
    metadata_check(metadata_keys(entries, &["prefill", "decode"]), "entries")?;
    for kind in ["prefill", "decode"] {
        let entry = &entries[kind];
        metadata_check(
            metadata_keys(entry, &["bytes", "sha256"])
                && metadata_integer(&entry["bytes"], 1)
                && metadata_digest(&entry["sha256"]),
            "entries",
        )?;
    }

    let weights = &obj["shared_weights"];
    metadata_check(
        metadata_keys(weights, &["identity", "bytes", "sha256"])
            && metadata_integer(&weights["bytes"], 1)
            && metadata_digest(&weights["identity"])
            && metadata_digest(&weights["sha256"]),
        "shared_weights",
    )?;

    let tokenizer = obj["tokenizer"].as_object();
    metadata_check(tokenizer.is_some(), "tokenizer")?;
    let tokenizer = tokenizer.unwrap();
    for required in TOKENIZER_REQUIRED_FILES {
        metadata_check(tokenizer.contains_key(required), "tokenizer")?;
    }
    for (name, entry) in tokenizer {
        let allowed = TOKENIZER_REQUIRED_FILES.contains(&name.as_str())
            || TOKENIZER_OPTIONAL_FILES.contains(&name.as_str());
        metadata_check(
            allowed
                && metadata_keys(entry, &["bytes", "sha256"])
                && metadata_integer(&entry["bytes"], 1)
                && metadata_digest(&entry["sha256"]),
            "tokenizer",
        )?;
    }

    Ok(())
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StateSpec {
    pub name: String,
    pub kind: String,
    pub layer: i64,
    pub shape: Vec<i64>,
    pub stride: Vec<i64>,
    pub dtype: DType,
}

impl StateSpec {
    pub fn is_kv(&self) -> bool {
        self.kind == "key" || self.kind == "value"
    }
}

/// Validated package contract shared by CLI, validation, and benchmark runners.
#[derive(Clone, Debug)]
pub struct Spec {
    pub model_type: ModelType,
    pub capacity: i64,
    pub max_prefill: i64,
    pub vocab: i64,
    pub max_batch_size: i64,
    pub eos: Vec<i64>,
    pub compute_capability: [i64; 2],
    pub metadata: Value,
    pub shared_weights: Value,
    pub state: Vec<StateSpec>,
}

impl Spec {
    pub fn load_base(directory: &Path) -> Result<Self> {
        let metadata = read_json(&directory.join(METADATA_FILE), "Cannot open metadata: ")?;
        validate_metadata(&metadata)?;

        let model_type = ModelType::parse(metadata["model_type"].as_str().unwrap())?;
        let capacity = metadata["cache_length"].as_i64().unwrap();
        let max_batch_size = metadata["state"][0]["shape"][0].as_i64().unwrap();
        let max_prefill = metadata["max_prefill_tokens"].as_i64().unwrap();
        let vocab = metadata["vocab_size"].as_i64().unwrap();
        let eos: Vec<i64> = metadata["eos_token_ids"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_i64().unwrap())
            .collect();
        let sm_arr = metadata["cuda_compute_capability"].as_array().unwrap();
        let compute_capability = [sm_arr[0].as_i64().unwrap(), sm_arr[1].as_i64().unwrap()];
        let shared_weights = metadata["shared_weights"].clone();

        // Match the linked LibTorch version; non-`git` `+` suffixes must match `cu<major><minor>`.
        let build = torch_build_info();
        let version = metadata["torch_version"].as_str().unwrap();
        match version.split_once('+') {
            None => require(version == build.torch_version, "Torch version mismatch")?,
            Some((base, suffix)) => {
                require(base == build.torch_version, "Torch version mismatch")?;
                if !suffix.starts_with("git") {
                    let expected = format!(
                        "cu{}{}",
                        build.cudart_version / 1000,
                        (build.cudart_version % 1000) / 10
                    );
                    require(suffix == expected, "Torch version mismatch")?;
                }
            }
        }
        require(
            metadata["cxx11_abi"].as_bool() == Some(build.cxx11_abi),
            "C++ ABI mismatch",
        )?;

        let mut state = Vec::new();
        for item in metadata["state"].as_array().unwrap() {
            let kind = item["kind"].as_str().unwrap().to_string();
            let layer = item["layer"].as_i64().unwrap();
            let shape: Vec<i64> = item["shape"]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_i64().unwrap())
                .collect();
            let stride = contiguous_strides(&shape);
            let dtype = if item["dtype"].as_str() == Some("float32") {
                DType::Float32
            } else {
                DType::BFloat16
            };
            state.push(StateSpec {
                name: format!("layer_{layer}.{kind}"),
                kind,
                layer,
                shape,
                stride,
                dtype,
            });
        }

        Ok(Self {
            model_type,
            capacity,
            max_prefill,
            vocab,
            max_batch_size,
            eos,
            compute_capability,
            metadata,
            shared_weights,
            state,
        })
    }

    /// Parses `metadata.json`, validates the 13-field schema, and checks model-specific state geometry.
    pub fn load(directory: &Path) -> Result<Self> {
        let spec = Self::load_base(directory)?;
        spec.validate_model_state()?;
        Ok(spec)
    }

    fn validate_model_state(&self) -> Result<()> {
        match self.model_type {
            ModelType::Qwen3 => {
                metadata_check(
                    !self.state.is_empty() && self.state.len().is_multiple_of(2),
                    "state",
                )?;
                for (pair_idx, pair) in self.state.as_chunks::<2>().0.iter().enumerate() {
                    let (key, value) = (&pair[0], &pair[1]);
                    let expected_layer = pair_idx as i64;
                    metadata_check(
                        key.layer == expected_layer
                            && value.layer == expected_layer
                            && key.kind == "key"
                            && value.kind == "value"
                            && key.dtype == DType::BFloat16
                            && value.dtype == DType::BFloat16,
                        "state",
                    )?;
                    require(key.shape == value.shape, "K/V shapes differ")?;
                }
            }
            ModelType::Qwen35 | ModelType::Qwen38 => {
                require(
                    !self.state.is_empty() && self.state.len().is_multiple_of(2),
                    "Invalid state count",
                )?;
                for (pair_idx, pair) in self.state.as_chunks::<2>().0.iter().enumerate() {
                    let (first, second) = (&pair[0], &pair[1]);
                    let linear = first.kind == "conv";
                    let expected_layer = pair_idx as i64;
                    require(
                        (linear || first.kind == "key")
                            && second.kind == if linear { "recurrent" } else { "value" }
                            && first.layer == expected_layer
                            && second.layer == expected_layer
                            && first.dtype == DType::BFloat16
                            && second.dtype
                                == if linear {
                                    DType::Float32
                                } else {
                                    DType::BFloat16
                                },
                        "Invalid state order or dtype",
                    )?;
                    if !linear {
                        require(first.shape == second.shape, "K/V shapes differ")?;
                    }
                }
            }
        }
        Ok(())
    }

    pub fn validate_files(&self, directory: &Path) -> Result<ValidatedBundle> {
        let bundle = ValidatedBundle::verify(directory, &self.metadata)?;
        require(
            ModelType::from_identity(bundle.identity())? == self.model_type,
            "Package model identity mismatch",
        )?;
        Ok(bundle)
    }

    /// Returns the checkpoint model ID declared in `metadata.json` (`checkpoint.model`).
    pub fn checkpoint_model(&self) -> &str {
        self.metadata["checkpoint"]["model"].as_str().unwrap_or("")
    }

    pub fn check_token_ids(&self, ids: &[i64]) -> Result<()> {
        for &id in ids {
            require_kind(
                id >= 0 && id < self.vocab,
                ErrorKind::InvalidInput,
                "Token ID out of range",
            )?;
        }
        Ok(())
    }

    pub fn check_tokens(&self, ids: &[i64]) -> Result<()> {
        require_kind(
            !ids.is_empty(),
            ErrorKind::InvalidInput,
            format!("Expected 1..{} input tokens", self.capacity),
        )?;
        require_kind(
            (ids.len() as i64) <= self.capacity,
            ErrorKind::CapacityExceeded,
            format!("Expected 1..{} input tokens", self.capacity),
        )?;
        self.check_token_ids(ids)
    }

    pub fn check_batch_tokens(&self, rows: &[&[i64]]) -> Result<i64> {
        require_kind(
            !rows.is_empty() && (rows.len() as i64) <= self.max_batch_size,
            ErrorKind::InvalidInput,
            "Batch size outside package range",
        )?;
        let mut length: usize = 0;
        for row in rows {
            require_kind(
                row.len() <= self.max_prefill as usize,
                ErrorKind::InvalidInput,
                "Too many tokens per compiled call",
            )?;
            self.check_token_ids(row)?;
            length = length.max(row.len());
        }
        require_kind(
            length > 0,
            ErrorKind::InvalidInput,
            "Expected at least one active row",
        )?;
        Ok(length as i64)
    }

    pub fn check_request(&self, ids: &[i64], limit: i64) -> Result<()> {
        self.check_tokens(ids)?;
        require_kind(
            limit > 0,
            ErrorKind::InvalidInput,
            "Generation budget must be positive and prompt + budget must fit the context",
        )?;
        require_kind(
            limit <= self.capacity - (ids.len() as i64),
            ErrorKind::CapacityExceeded,
            "Generation budget must be positive and prompt + budget must fit the context",
        )
    }

    pub fn read_ids(&self, path: &Path) -> Result<Vec<i64>> {
        let text = fs::read_to_string(path)
            .map_err(|_| Error::new(format!("Cannot open token IDs: {}", path.display())))?;
        let mut ids = Vec::new();
        for word in text.split_ascii_whitespace() {
            ids.push(parse_integer(word)?);
            require(
                (ids.len() as i64) <= self.capacity,
                format!("Expected 1..{} input tokens", self.capacity),
            )?;
        }
        self.check_tokens(&ids)?;
        Ok(ids)
    }
}

/// Verifies CUDA device 0 is available and its SM capability matches `metadata.json`.
pub fn checked_device(spec: &Spec) -> Result<Device> {
    let (major, minor) = cuda_device_capability(0)?;
    require(
        spec.compute_capability == [major, minor],
        "GPU architecture mismatch",
    )?;
    Ok(Device::Cuda(0))
}

/// Applies deterministic runtime settings (1 CPU thread, TF32 off) and verifies CUDA device 0.
pub fn configured_device(spec: &Spec) -> Result<Device> {
    // SAFETY: `xinfer_shim_configure_runtime` is thread-safe/noexcept and writes `err` on failure.
    shim_call(|err| unsafe { torch_sys::xinfer_shim_configure_runtime(err) })?;
    checked_device(spec)
}
