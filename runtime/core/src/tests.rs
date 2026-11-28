use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

use serde_json::{json, Value};

use crate::{
    check_view_bounds, contiguous_strides, generate_with, parse_integer, prefill_with,
    read_safetensors_metadata, sha256_file, torch_build_info, validate_metadata, DType, Device,
    Error, ErrorKind, Model, ModelType, Result, SharedWeights, Spec, StatePolicy, StateSpec,
    StepDriver, Tensor, ValidatedBundle, MAX_BATCH_SIZE, MAX_CACHE_LENGTH, METADATA_FILE,
    SHARED_WEIGHTS_FILE, SHARED_WEIGHTS_MANIFEST_KEY,
};

static COUNTER: AtomicU64 = AtomicU64::new(0);

struct TempDir {
    path: PathBuf,
}

impl TempDir {
    fn new(tag: &str) -> Self {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let seq = COUNTER.fetch_add(1, Ordering::Relaxed);
        let path = std::env::temp_dir().join(format!(
            "xinfer-rust-core-{tag}-{}-{nanos}-{seq}",
            std::process::id()
        ));
        fs::create_dir_all(&path).unwrap();
        Self { path }
    }

    fn path(&self) -> &Path {
        &self.path
    }
}

impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.path);
    }
}

fn state_descriptor(kind: &str, batch: i64) -> StateSpec {
    let shape = if kind == "conv" {
        vec![batch, 2, 3]
    } else {
        vec![batch, 2, 3, 4]
    };
    let stride = contiguous_strides(&shape);
    StateSpec {
        name: format!("layer_0.{kind}"),
        kind: kind.to_string(),
        layer: 0,
        shape,
        stride,
        dtype: if kind == "recurrent" {
            DType::Float32
        } else {
            DType::BFloat16
        },
    }
}

fn kv_specs(batch: i64) -> Vec<StateSpec> {
    vec![
        state_descriptor("key", batch),
        state_descriptor("value", batch),
    ]
}

fn mixed_specs(batch: i64) -> Vec<StateSpec> {
    vec![
        state_descriptor("conv", batch),
        state_descriptor("recurrent", batch),
        state_descriptor("key", batch),
        state_descriptor("value", batch),
    ]
}

fn tensor_bytes_cpu(t: &Tensor) -> Vec<u8> {
    let nbytes = (t.numel().unwrap() as usize) * t.dtype().unwrap().element_size();
    let ptr = t.data_ptr().unwrap().cast::<u8>();
    unsafe { std::slice::from_raw_parts(ptr, nbytes) }.to_vec()
}

fn tensor_f32_values(t: &Tensor) -> Vec<f32> {
    match t.dtype().unwrap() {
        DType::BFloat16 => t.bf16_to_f32_host().unwrap(),
        DType::Float32 => {
            let n = t.numel().unwrap() as usize;
            let ptr = t.data_ptr().unwrap().cast::<f32>();
            unsafe { std::slice::from_raw_parts(ptr, n) }.to_vec()
        }
        other => panic!("unexpected dtype: {other:?}"),
    }
}

fn base_metadata(capacity: i64, model_type: &str) -> Value {
    let build = torch_build_info();
    json!({
        "model_type": model_type,
        "torch_version": build.torch_version,
        "cxx11_abi": build.cxx11_abi,
        "cuda_compute_capability": [8, 9],
        "cache_length": capacity,
        "max_prefill_tokens": 512,
        "vocab_size": 16,
        "eos_token_ids": [1],
        "checkpoint": { "model": "test", "revision": "test" },
        "state": [
            { "layer": 0, "kind": "key", "shape": [8, 8, capacity, 128], "dtype": "bfloat16" },
            { "layer": 0, "kind": "value", "shape": [8, 8, capacity, 128], "dtype": "bfloat16" },
        ],
        "entries": {
            "prefill": { "bytes": 1, "sha256": "a".repeat(64) },
            "decode": { "bytes": 1, "sha256": "a".repeat(64) },
        },
        "shared_weights": {
            "identity": "b".repeat(64),
            "bytes": 1,
            "sha256": "c".repeat(64),
        },
        "tokenizer": {
            "tokenizer.json": { "bytes": 1, "sha256": "d".repeat(64) },
            "tokenizer_config.json": { "bytes": 1, "sha256": "e".repeat(64) },
        },
    })
}

fn write_metadata(dir: &Path, metadata: &Value) {
    fs::write(dir.join(METADATA_FILE), metadata.to_string()).unwrap();
}

// ============================================================================
// StatePolicy contract tests (ported from runtime/common/tests/state.cpp)
// ============================================================================

#[test]
fn state_qwen3_logits_only() {
    let mut state = StatePolicy::new(kv_specs(1), Device::Cpu, None).unwrap();
    assert!(state.accept(1, 1).is_err());

    state.reset(7.0).unwrap();
    let ptrs_before: Vec<usize> = state
        .tensors()
        .iter()
        .map(|t| t.data_ptr().unwrap() as usize)
        .collect();
    for &ptr in &ptrs_before {
        assert_eq!(ptr & 15, 0, "KV state buffer is not 16-byte aligned");
    }
    let bytes_before: Vec<Vec<u8>> = state.tensors().iter().map(tensor_bytes_cpu).collect();
    state.accept(1, 1).unwrap();
    for (i, t) in state.tensors().iter().enumerate() {
        assert_eq!(t.data_ptr().unwrap() as usize, ptrs_before[i]);
        assert_eq!(tensor_bytes_cpu(t), bytes_before[i]);
    }

    state.reset(-4.0).unwrap();
    for (i, t) in state.tensors().iter().enumerate() {
        assert_eq!(t.data_ptr().unwrap() as usize, ptrs_before[i]);
        assert!(tensor_f32_values(t).iter().all(|&v| v == -4.0));
    }

    assert!(state.accept(0, 1).is_err());
    assert!(state.accept(2, 1).is_err());
}

#[test]
fn state_qwen35_inplace_and_reset() {
    let mut state = StatePolicy::new(mixed_specs(1), Device::Cpu, None).unwrap();
    state.reset(7.0).unwrap();
    let ptrs_before: Vec<usize> = state
        .tensors()
        .iter()
        .map(|t| t.data_ptr().unwrap() as usize)
        .collect();
    for (i, t) in state.tensors().iter().enumerate() {
        let kv = i >= 2;
        assert_eq!(t.data_ptr().unwrap() as usize & 15, 0);
        let expected = if kv { 7.0 } else { 0.0 };
        assert!(tensor_f32_values(t).iter().all(|&v| v == expected));
    }

    state.tensors_mut()[0].fill_scalar(2.0).unwrap();
    state.tensors_mut()[1].fill_scalar(3.0).unwrap();
    state.accept(1, 1).unwrap();
    for (i, t) in state.tensors().iter().enumerate() {
        assert_eq!(t.data_ptr().unwrap() as usize, ptrs_before[i]);
    }
    assert!(tensor_f32_values(&state.tensors()[0])
        .iter()
        .all(|&v| v == 2.0));
    assert!(tensor_f32_values(&state.tensors()[1])
        .iter()
        .all(|&v| v == 3.0));

    state.reset(-4.0).unwrap();
    for (i, t) in state.tensors().iter().enumerate() {
        let kv = i >= 2;
        assert_eq!(t.data_ptr().unwrap() as usize, ptrs_before[i]);
        let expected = if kv { -4.0 } else { 0.0 };
        assert!(tensor_f32_values(t).iter().all(|&v| v == expected));
    }
}

#[test]
fn state_allocated_batch_size_bounds() {
    let mut state = StatePolicy::new(mixed_specs(8), Device::Cpu, Some(3)).unwrap();
    assert_eq!(state.allocated_batch_size(), 3);
    state.reset(5.0).unwrap();
    for t in state.tensors() {
        assert_eq!(t.sizes().unwrap()[0], 3);
    }
    state.accept(1, 1).unwrap();
    state.accept(1, 3).unwrap();
    assert!(state.accept(1, 0).is_err());
    assert!(state.accept(1, 4).is_err());

    assert!(StatePolicy::new(mixed_specs(8), Device::Cpu, Some(0)).is_err());
    assert!(StatePolicy::new(mixed_specs(8), Device::Cpu, Some(9)).is_err());
}

#[test]
fn state_rejects_bad_descriptors() {
    assert!(StatePolicy::new(vec![], Device::Cpu, None).is_err());

    let mut empty_shape = mixed_specs(1);
    empty_shape[0].shape.clear();
    assert!(StatePolicy::new(empty_shape, Device::Cpu, None).is_err());

    let mut zero_batch = mixed_specs(1);
    zero_batch[0].shape[0] = 0;
    assert!(StatePolicy::new(zero_batch, Device::Cpu, None).is_err());

    let mut inconsistent = mixed_specs(1);
    inconsistent[3].shape[0] = 2;
    assert!(StatePolicy::new(inconsistent, Device::Cpu, None).is_err());
}

// ============================================================================
// Metadata, Spec, and ModelType tests (ported from generation.cpp)
// ============================================================================

#[test]
fn qwen38_identity_uses_shared_qwen35_state_rules() {
    for kind in ["qwen35", "qwen38"] {
        let dir = TempDir::new(kind);
        let mut meta = base_metadata(512, kind);
        meta["state"] = json!([
            { "layer": 0, "kind": "conv", "shape": [8, 4, 3], "dtype": "bfloat16" },
            { "layer": 0, "kind": "recurrent", "shape": [8, 2, 4, 4], "dtype": "float32" },
            { "layer": 1, "kind": "key", "shape": [8, 4, 512, 64], "dtype": "bfloat16" },
            { "layer": 1, "kind": "value", "shape": [8, 4, 512, 64], "dtype": "bfloat16" },
        ]);
        write_metadata(dir.path(), &meta);

        let model = Model::open(dir.path()).unwrap();
        assert_eq!(model.spec().model_type.as_str(), kind);
        assert_eq!(ModelType::parse(kind).unwrap(), model.spec().model_type);

        let id_str = if kind == "qwen38" {
            "qwen3.8"
        } else {
            "qwen3.5"
        };
        let other_str = if kind == "qwen38" {
            "qwen3.5"
        } else {
            "qwen3.8"
        };
        assert_eq!(
            ModelType::from_identity(&json!({ "model": id_str })).unwrap(),
            model.spec().model_type
        );
        assert_ne!(
            ModelType::from_identity(&json!({ "model": other_str })).unwrap(),
            model.spec().model_type
        );
    }
}

#[test]
fn context_contract_and_retired_fields() {
    let metadata = base_metadata(16384, "qwen3");
    validate_metadata(&metadata).unwrap();

    let mut small = metadata.clone();
    small["cache_length"] = json!(511);
    assert_eq!(
        validate_metadata(&small).unwrap_err().to_string(),
        "Unsupported package metadata: cache_length"
    );

    small["cache_length"] = json!(512);
    for state in small["state"].as_array_mut().unwrap() {
        state["shape"][2] = json!(512);
    }
    validate_metadata(&small).unwrap();

    let mut too_large = small.clone();
    too_large["cache_length"] = json!(MAX_CACHE_LENGTH + 1);
    for state in too_large["state"].as_array_mut().unwrap() {
        state["shape"][2] = json!(MAX_CACHE_LENGTH + 1);
    }
    assert_eq!(
        validate_metadata(&too_large).unwrap_err().to_string(),
        "Unsupported package metadata: cache_length"
    );

    too_large["cache_length"] = json!(MAX_CACHE_LENGTH);
    for state in too_large["state"].as_array_mut().unwrap() {
        state["shape"][2] = json!(MAX_CACHE_LENGTH);
    }
    validate_metadata(&too_large).unwrap();

    let mut bad_prefill = metadata.clone();
    bad_prefill["max_prefill_tokens"] = json!(1024);
    assert_eq!(
        validate_metadata(&bad_prefill).unwrap_err().to_string(),
        "Unsupported package metadata: max_prefill_tokens"
    );

    // Non-integer float values must be rejected strictly.
    let mut float_vocab = metadata.clone();
    float_vocab["vocab_size"] = json!(16.0);
    assert_eq!(
        validate_metadata(&float_vocab).unwrap_err().to_string(),
        "Unsupported package metadata: vocab_size"
    );

    for field in [
        "max_prompt_length",
        "entry_point",
        "token_range",
        "export_options",
        "tf32",
        "triton_enable_fp_fusion",
        "inductor_configs",
        "package_bytes",
        "package_sha256",
        "model_specific",
        "eager_numerics",
    ] {
        let mut retired = metadata.clone();
        retired[field] = json!(1);
        assert_eq!(
            validate_metadata(&retired).unwrap_err().to_string(),
            format!("Unsupported package metadata: {field}")
        );
    }
}

#[test]
fn spec_checks_torch_cuda_suffix() {
    let dir = TempDir::new("torch-ver");
    let build = torch_build_info();
    let cuda_tag = format!(
        "cu{}{}",
        build.cudart_version / 1000,
        (build.cudart_version % 1000) / 10
    );

    for accepted in [
        build.torch_version.clone(),
        format!("{}+{cuda_tag}", build.torch_version),
        format!("{}+gitabcdef", build.torch_version),
    ] {
        let mut meta = base_metadata(16384, "qwen3");
        meta["torch_version"] = json!(accepted);
        write_metadata(dir.path(), &meta);
        let spec = Spec::load(dir.path()).unwrap();
        assert_eq!(spec.capacity, 16384);
    }

    for bad in [
        format!("{}+cu999", build.torch_version),
        format!("{}+cpu", build.torch_version),
        format!("{}+", build.torch_version),
        format!("0.0.0+{cuda_tag}"),
    ] {
        let mut meta = base_metadata(16384, "qwen3");
        meta["torch_version"] = json!(bad);
        write_metadata(dir.path(), &meta);
        assert_eq!(
            Spec::load(dir.path()).unwrap_err().to_string(),
            "Torch version mismatch"
        );
    }
}

#[test]
fn qwen3_spec_accepts_general_kv_shapes_and_size_free_identity() {
    assert_eq!(
        ModelType::from_identity(&json!({ "model": "qwen3" })).unwrap(),
        ModelType::Qwen3
    );
    assert_eq!(
        ModelType::from_identity(&json!({ "model": "qwen3.5" })).unwrap(),
        ModelType::Qwen35
    );
    assert!(ModelType::from_identity(&json!({ "model": "qwen3-0.6b" })).is_err());

    let dir = TempDir::new("qwen3-geom");
    for shape in [
        json!([8, 4, 16384, 128]),
        json!([8, 8, 16384, 64]),
        json!([8, 8, 16384, 256]),
    ] {
        let mut meta = base_metadata(16384, "qwen3");
        for item in meta["state"].as_array_mut().unwrap() {
            item["shape"] = shape.clone();
        }
        write_metadata(dir.path(), &meta);
        let spec = Spec::load(dir.path()).unwrap();
        assert_eq!(spec.state.len(), 2);
        assert_eq!(spec.state[0].shape, spec.state[1].shape);
    }

    let mut meta = base_metadata(16384, "qwen3");
    meta["state"][0]["shape"] = json!([8, 4, 16384, 128]);
    meta["state"][1]["shape"] = json!([8, 8, 16384, 128]);
    write_metadata(dir.path(), &meta);
    assert_eq!(
        Spec::load(dir.path()).unwrap_err().to_string(),
        "K/V shapes differ"
    );
}

#[test]
fn parse_integer_matches_from_chars_contract() {
    assert_eq!(parse_integer("0").unwrap(), 0);
    assert_eq!(parse_integer("-42").unwrap(), -42);
    assert_eq!(parse_integer("9223372036854775807").unwrap(), i64::MAX);
    assert!(parse_integer("+1").is_err());
    assert!(parse_integer("1.5").is_err());
    assert!(parse_integer("").is_err());
    assert!(parse_integer("999999999999999999999999999999").is_err());
}

// ============================================================================
// SHA-256 tests (NIST vectors + 256 KiB streaming boundary)
// ============================================================================

#[test]
fn sha256_matches_nist_vectors_and_buffer_boundaries() {
    let dir = TempDir::new("sha256");
    let vectors: &[(&[u8], &str)] = &[
        (
            b"",
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
        ),
        (
            b"abc",
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
        ),
        (
            b"abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq",
            "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1",
        ),
    ];
    for (i, (payload, expected)) in vectors.iter().enumerate() {
        let path = dir.path().join(format!("vec-{i}.bin"));
        fs::write(&path, payload).unwrap();
        assert_eq!(sha256_file(&path).unwrap(), *expected);
    }

    // 1_000_000 'a' bytes (standard NIST long vector, spans 4 x 256 KiB chunks).
    let million_a = dir.path().join("million-a.bin");
    fs::write(&million_a, vec![b'a'; 1_000_000]).unwrap();
    assert_eq!(
        sha256_file(&million_a).unwrap(),
        "cdc76e5c9914fb9281a1c7e284d73e67f1809a48a497200e046d39ccc7112cd0"
    );
}

// ============================================================================
// Safetensors / SharedWeights CPU loading & malformed input tests
// ============================================================================

#[test]
fn shared_weights_cpu_load_including_fp8_and_views() {
    use safetensors::tensor::{Dtype as StDtype, TensorView};

    let dir = TempDir::new("weights-cpu");
    let prefill_path = dir.path().join("prefill.pt2");
    let decode_path = dir.path().join("decode.pt2");
    fs::write(&prefill_path, b"prefill-stub").unwrap();
    fs::write(&decode_path, b"decode-stub").unwrap();
    let prefill_sha = sha256_file(&prefill_path).unwrap();
    let decode_sha = sha256_file(&decode_path).unwrap();
    let identity_sha = "d".repeat(64);

    // Build two CPU storages: one FP8 (F8_E4M3, 6 bytes) and one F32 (4 floats = 16 bytes).
    let fp8_bytes: [u8; 6] = [0x38, 0x3c, 0x40, 0x42, 0x44, 0x46];
    let f32_vals: [f32; 4] = [1.25, -2.5, 3.75, 4.0];
    let f32_bytes: Vec<u8> = f32_vals.iter().flat_map(|v| v.to_le_bytes()).collect();

    let manifest = json!({
        "identity": { "model": "qwen3" },
        "identity_sha256": identity_sha,
        "entries": {
            "prefill": { "package_sha256": prefill_sha, "constants": { "w_fp8": "t_fp8", "w_f32": "t_f32" } },
            "decode":  { "package_sha256": decode_sha,  "constants": { "w_fp8": "t_fp8", "w_f32": "t_f32" } },
        },
        "sources": {},
        "storages": {
            "s_fp8": { "dtype": "float8_e4m3fn", "device": "cpu", "bytes": 6 },
            "s_f32": { "dtype": "float32",       "device": "cpu", "bytes": 16 },
        },
        "tensors": {
            "t_fp8": { "storage": "s_fp8", "dtype": "float8_e4m3fn", "device": "cpu", "shape": [2, 2], "stride": [2, 1], "offset": 2 },
            "t_f32": { "storage": "s_f32", "dtype": "float32",       "device": "cpu", "shape": [2, 2], "stride": [2, 1], "offset": 0 },
        },
    });
    let st_tensors = [
        (
            "s_fp8",
            TensorView::new(StDtype::F8_E4M3, vec![6], &fp8_bytes).unwrap(),
        ),
        (
            "s_f32",
            TensorView::new(StDtype::F32, vec![4], &f32_bytes).unwrap(),
        ),
    ];
    let st_meta = [(
        SHARED_WEIGHTS_MANIFEST_KEY.to_string(),
        manifest.to_string(),
    )]
    .into_iter()
    .collect();
    let serialized = safetensors::serialize(st_tensors, Some(st_meta)).unwrap();
    let weights_path = dir.path().join(SHARED_WEIGHTS_FILE);
    fs::write(&weights_path, serialized).unwrap();
    let weights_sha = sha256_file(&weights_path).unwrap();
    let weights_bytes = fs::metadata(&weights_path).unwrap().len();

    let tok_json = dir.path().join("tokenizer.json");
    let tok_cfg = dir.path().join("tokenizer_config.json");
    fs::write(&tok_json, b"{}").unwrap();
    fs::write(&tok_cfg, b"{}").unwrap();
    let tok_json_sha = sha256_file(&tok_json).unwrap();
    let tok_cfg_sha = sha256_file(&tok_cfg).unwrap();

    let mut meta = base_metadata(512, "qwen3");
    meta["entries"]["prefill"] = json!({ "bytes": 12, "sha256": prefill_sha });
    meta["entries"]["decode"] = json!({ "bytes": 11, "sha256": decode_sha });
    meta["shared_weights"] = json!({
        "identity": identity_sha,
        "bytes": weights_bytes,
        "sha256": weights_sha,
    });
    meta["tokenizer"] = json!({
        "tokenizer.json": { "bytes": 2, "sha256": tok_json_sha },
        "tokenizer_config.json": { "bytes": 2, "sha256": tok_cfg_sha },
    });
    write_metadata(dir.path(), &meta);

    let spec = Spec::load(dir.path()).unwrap();
    let bundle = spec.validate_files(dir.path()).unwrap();
    let weights = SharedWeights::load(bundle, &spec.shared_weights).unwrap();

    let fp8 = weights.tensor("t_fp8").unwrap();
    assert_eq!(fp8.dtype().unwrap(), DType::Float8E4M3Fn);
    assert_eq!(fp8.sizes().unwrap(), vec![2, 2]);
    assert_eq!(tensor_bytes_cpu(fp8), vec![0x40, 0x42, 0x44, 0x46]);

    let f32_t = weights.tensor("t_f32").unwrap();
    assert_eq!(f32_t.dtype().unwrap(), DType::Float32);
    assert_eq!(tensor_f32_values(f32_t), vec![1.25, -2.5, 3.75, 4.0]);

    // Check view bounds rejection cases.
    assert!(check_view_bounds(&[2, 2], &[2, 1], 2, 6).is_ok());
    assert!(check_view_bounds(&[2, 2], &[2, 1], 3, 6).is_err());
    assert!(check_view_bounds(&[0, 4], &[4, 1], 6, 6).is_ok());
    assert!(check_view_bounds(&[0, 4], &[4, 1], 7, 6).is_err());
    assert!(check_view_bounds(&[i64::MAX, 2], &[1, i64::MAX], 0, i64::MAX).is_err());

    // Check truncated safetensors header rejection.
    let bad_st = dir.path().join("bad.safetensors");
    fs::write(&bad_st, b"short").unwrap();
    assert_eq!(
        read_safetensors_metadata(&bad_st).unwrap_err().to_string(),
        "Truncated safetensors header"
    );

    // Check mismatched contracts rejected.
    let bundle2 = ValidatedBundle::verify(dir.path(), &meta).unwrap();
    let mut other_contract = spec.shared_weights.clone();
    other_contract["bytes"] = json!(999);
    assert_eq!(
        SharedWeights::load(bundle2, &other_contract)
            .err()
            .unwrap()
            .to_string(),
        "Shared weights contracts differ"
    );
}

// ============================================================================
// Chunking, EOS, budget, and no-mutation tests (ported from generation.cpp)
// ============================================================================

struct MockLogits {
    position: i64,
    vocab: i64,
}

#[derive(Clone, PartialEq, Eq, Debug)]
struct MockSnapshot {
    position: i64,
    resets: i64,
    calls: Vec<(i64, usize)>,
    consumed: Vec<i64>,
    selections: Vec<i64>,
}

struct MockRuntime {
    spec: Spec,
    position: i64,
    prefill_calls: i64,
    decode_calls: i64,
    resets: i64,
    calls: Vec<(i64, usize)>,
    consumed: Vec<i64>,
    selections: Vec<i64>,
}

impl MockRuntime {
    fn new(spec: Spec) -> Self {
        Self {
            spec,
            position: 0,
            prefill_calls: 0,
            decode_calls: 0,
            resets: 0,
            calls: Vec::new(),
            consumed: Vec::new(),
            selections: Vec::new(),
        }
    }

    fn snapshot(&self) -> MockSnapshot {
        MockSnapshot {
            position: self.position,
            resets: self.resets,
            calls: self.calls.clone(),
            consumed: self.consumed.clone(),
            selections: self.selections.clone(),
        }
    }

    fn generate_checked(
        &mut self,
        prompt: &[i64],
        limit: i64,
        trace: Option<&mut dyn Write>,
    ) -> Result<Vec<i64>> {
        self.spec.check_request(prompt, limit)?;
        let eos = self.spec.eos.clone();
        generate_with(self, prompt, limit, &eos, trace)
    }
}

impl StepDriver for MockRuntime {
    type Logits = MockLogits;

    fn reset(&mut self) -> Result<()> {
        self.resets += 1;
        self.position = 0;
        self.prefill_calls = 0;
        self.decode_calls = 0;
        self.calls.clear();
        self.consumed.clear();
        self.selections.clear();
        Ok(())
    }

    fn step(&mut self, tokens: &[i64]) -> Result<MockLogits> {
        self.spec.check_batch_tokens(&[tokens])?;
        if (tokens.len() as i64) > self.spec.capacity - self.position {
            return Err(Error::new("Context capacity exceeded"));
        }
        self.calls.push((self.position, tokens.len()));
        self.consumed.extend_from_slice(tokens);
        self.position += tokens.len() as i64;
        if tokens.len() == 1 {
            self.decode_calls += 1;
        } else {
            self.prefill_calls += 1;
        }
        Ok(MockLogits {
            position: self.position,
            vocab: self.spec.vocab,
        })
    }

    fn prefill(&mut self, prompt: &[i64]) -> Result<MockLogits> {
        self.spec.check_tokens(prompt)?;
        if (prompt.len() as i64) > self.spec.capacity - self.position {
            return Err(Error::new("Context capacity exceeded"));
        }
        let max_prefill = self.spec.max_prefill;
        prefill_with(self, prompt, max_prefill)
    }

    fn argmax(&mut self, logits: &MockLogits) -> Result<i64> {
        self.selections.push(logits.position);
        Ok(logits.position % 7)
    }

    fn write_trace(&mut self, logits: &MockLogits, writer: &mut dyn Write) -> Result<()> {
        let mut row = vec![0.0f32; logits.vocab as usize];
        row[(logits.position % 7) as usize] = 1.0;
        for val in row {
            writer
                .write_all(&val.to_le_bytes())
                .map_err(|_| Error::new("Failed to write logits trace"))?;
        }
        Ok(())
    }
}

#[test]
fn prefill_chunking_boundaries_and_final_singleton_decode() {
    let dir = TempDir::new("chunking");
    write_metadata(dir.path(), &base_metadata(16384, "qwen3"));
    let spec = Spec::load(dir.path()).unwrap();
    let mut rt = MockRuntime::new(spec);

    let cases: &[(usize, Vec<usize>, i64, i64)] = &[
        (1, vec![1], 0, 1),
        (511, vec![511], 1, 0),
        (512, vec![512], 1, 0),
        (513, vec![512, 1], 1, 1),
        (8191, [vec![512; 15], vec![511]].concat(), 16, 0),
        (8192, vec![512; 16], 16, 0),
    ];

    for (len, expected_chunks, expected_prefill, expected_decode) in cases {
        rt.reset().unwrap();
        let prompt: Vec<i64> = (0..*len).map(|i| (i % 16) as i64).collect();
        let logits = rt.prefill(&prompt).unwrap();
        assert_eq!(logits.position, *len as i64);
        assert_eq!(rt.prefill_calls, *expected_prefill);
        assert_eq!(rt.decode_calls, *expected_decode);
        assert_eq!(rt.consumed, prompt);
        let actual_chunks: Vec<usize> = rt.calls.iter().map(|&(_, c)| c).collect();
        assert_eq!(actual_chunks, *expected_chunks);
    }
}

#[test]
fn generate_stops_at_eos_and_limit_and_rejects_without_mutation() {
    let dir = TempDir::new("gen-contract");
    write_metadata(dir.path(), &base_metadata(16384, "qwen3"));
    let spec = Spec::load(dir.path()).unwrap();
    let mut rt = MockRuntime::new(spec);

    // Prompt length 5 -> positions after prefill = 5, 6, 7, 8 (8 % 7 == 1 == EOS).
    let mut trace = Vec::new();
    let generated = rt
        .generate_checked(&[2, 3, 4, 5, 6], 10, Some(&mut trace))
        .unwrap();
    assert_eq!(generated, vec![5, 6, 0, 1]);
    assert_eq!(trace.len(), 4 * 16 * 4);
    assert_eq!(rt.selections, vec![5, 6, 7, 8]);

    // Budget limit 2 stops before EOS.
    let limited = rt.generate_checked(&[2, 3, 4, 5, 6], 2, None).unwrap();
    assert_eq!(limited, vec![5, 6]);

    // Rejections leave runtime snapshot completely untouched.
    let snap = rt.snapshot();
    assert!(rt.generate_checked(&[], 1, None).is_err());
    assert_eq!(rt.snapshot(), snap);

    assert!(rt.generate_checked(&[2], 0, None).is_err());
    assert_eq!(rt.snapshot(), snap);

    let full_prompt = vec![2i64; 16384];
    assert!(rt.generate_checked(&full_prompt, 1, None).is_err());
    assert_eq!(rt.snapshot(), snap);

    let overflow_prefill = vec![2i64; 16385 - rt.position as usize];
    assert!(rt.prefill(&overflow_prefill).is_err());
    assert_eq!(rt.snapshot(), snap);

    assert!(rt.spec.check_batch_tokens(&[&[], &[]]).is_err());
    assert!(rt
        .spec
        .check_batch_tokens(&[&[0i64][..]; (MAX_BATCH_SIZE + 1) as usize])
        .is_err());
}

#[test]
fn rust_struct_drop_order_drops_borrowing_packages_before_weights() {
    use std::cell::RefCell;
    use std::rc::Rc;

    struct Tracked {
        label: &'static str,
        log: Rc<RefCell<Vec<&'static str>>>,
    }
    impl Drop for Tracked {
        fn drop(&mut self) {
            self.log.borrow_mut().push(self.label);
        }
    }
    struct MockRunner {
        _prefill: Tracked,
        _decode: Tracked,
        _weights: Tracked,
    }

    let log = Rc::new(RefCell::new(Vec::new()));
    {
        let _runner = MockRunner {
            _prefill: Tracked {
                label: "prefill",
                log: Rc::clone(&log),
            },
            _decode: Tracked {
                label: "decode",
                log: Rc::clone(&log),
            },
            _weights: Tracked {
                label: "weights",
                log: Rc::clone(&log),
            },
        };
    }
    assert_eq!(*log.borrow(), vec!["prefill", "decode", "weights"]);
}

#[test]
fn tensor_chained_narrow_and_slice_preserves_offset() {
    // Regression: aoti_torch__reinterpret_tensor adds its offset relative to self.storage_offset().
    let mut base = Tensor::empty_contiguous(&[8, 2, 6, 4], DType::Float32, Device::Cpu).unwrap();
    let total = (8 * 2 * 6 * 4) as usize;
    let host: Vec<f32> = (0..total).map(|i| i as f32).collect();
    let bytes: Vec<u8> = host.iter().flat_map(|v| v.to_le_bytes()).collect();
    base.copy_bytes_cpu(&bytes).unwrap();

    let row7 = base.narrow_dim0(7, 1).unwrap();
    assert_eq!(row7.storage_offset().unwrap(), 7 * 2 * 6 * 4);

    let sliced = row7.slice_dim(2, 2, 5).unwrap();
    assert_eq!(sliced.sizes().unwrap(), vec![1, 2, 3, 4]);
    assert_eq!(sliced.storage_offset().unwrap(), 7 * 2 * 6 * 4 + 2 * 4);

    let actual_bytes = sliced.to_contiguous_host_bytes().unwrap();
    let actual: Vec<f32> = actual_bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|&c| f32::from_le_bytes(c))
        .collect();
    let mut expected = Vec::new();
    for h in 0..2 {
        for s in 2..5 {
            for d in 0..4 {
                expected.push((7 * 48 + h * 24 + s * 4 + d) as f32);
            }
        }
    }
    assert_eq!(actual, expected);
}

#[test]
fn tokenizer_metadata_validation_and_file_checks() {
    let mut with_template = base_metadata(512, "qwen35");
    with_template["state"] = json!([
        { "layer": 0, "kind": "conv", "shape": [8, 6144, 4], "dtype": "bfloat16" },
        { "layer": 0, "kind": "recurrent", "shape": [8, 16, 128, 128], "dtype": "float32" },
    ]);
    with_template["tokenizer"]["chat_template.jinja"] =
        json!({ "bytes": 4, "sha256": "f".repeat(64) });
    validate_metadata(&with_template).unwrap();

    let mut missing_tok = base_metadata(512, "qwen3");
    missing_tok.as_object_mut().unwrap().remove("tokenizer");
    let err = validate_metadata(&missing_tok).unwrap_err();
    assert_eq!(err.kind(), ErrorKind::Unsupported);
    assert_eq!(err.to_string(), "Unsupported package metadata: tokenizer");

    let mut unknown_file = base_metadata(512, "qwen3");
    unknown_file["tokenizer"]["merges.txt"] = json!({ "bytes": 4, "sha256": "a".repeat(64) });
    assert_eq!(
        validate_metadata(&unknown_file).unwrap_err().to_string(),
        "Unsupported package metadata: tokenizer"
    );

    let mut bad_bytes = base_metadata(512, "qwen3");
    bad_bytes["tokenizer"]["tokenizer.json"]["bytes"] = json!(0);
    assert_eq!(
        validate_metadata(&bad_bytes).unwrap_err().to_string(),
        "Unsupported package metadata: tokenizer"
    );
}

#[test]
fn state_reset_slot_updates_only_target_row() {
    // Use 16-byte-aligned row strides matching production AOTI state descriptors.
    let aligned_spec = |kind: &str| {
        let shape = if kind == "conv" {
            vec![3, 2, 4]
        } else {
            vec![3, 2, 2, 4]
        };
        let stride = contiguous_strides(&shape);
        StateSpec {
            name: format!("layer_0.{kind}"),
            kind: kind.to_string(),
            layer: 0,
            shape,
            stride,
            dtype: if kind == "recurrent" {
                DType::Float32
            } else {
                DType::BFloat16
            },
        }
    };
    let specs = vec![
        aligned_spec("conv"),
        aligned_spec("recurrent"),
        aligned_spec("key"),
        aligned_spec("value"),
    ];
    let mut state = StatePolicy::new(specs, Device::Cpu, Some(3)).unwrap();
    state.reset(7.0).unwrap();

    // Write distinct non-zero values into every row of conv/recurrent and KV.
    for (idx, tensor) in state.tensors_mut().iter_mut().enumerate() {
        for slot in 0..3i64 {
            let mut row = tensor.narrow_dim0(slot, 1).unwrap();
            row.fill_scalar((slot + 1 + idx as i64) as f64).unwrap();
        }
    }

    // Resetting slot 1 with fill_kv=None zeroes conv/recurrent row 1 while leaving KV and rows 0,2 intact.
    state.reset_slot(1, None).unwrap();
    for (idx, tensor) in state.tensors().iter().enumerate() {
        let row0 = tensor_f32_values(&tensor.narrow_dim0(0, 1).unwrap());
        let row1 = tensor_f32_values(&tensor.narrow_dim0(1, 1).unwrap());
        let row2 = tensor_f32_values(&tensor.narrow_dim0(2, 1).unwrap());
        assert!(row0.iter().all(|&v| v == (1 + idx) as f32));
        assert!(row2.iter().all(|&v| v == (3 + idx) as f32));
        if idx < 2 {
            // conv, recurrent
            assert!(row1.iter().all(|&v| v == 0.0));
        } else {
            // key, value untouched when fill_kv is None
            assert!(row1.iter().all(|&v| v == (2 + idx) as f32));
        }
    }

    // Out-of-bounds slot fails with InvalidInput.
    let err = state.reset_slot(3, None).unwrap_err();
    assert_eq!(err.kind(), ErrorKind::InvalidInput);
}

#[test]
fn tensor_argmax_batch_selects_per_row_max_on_cpu() {
    let logits = Tensor::empty_contiguous(&[3, 5], DType::Float32, Device::Cpu).unwrap();
    let ptr = logits.data_ptr().unwrap().cast::<f32>();
    let values: [f32; 15] = [
        -1.0, 2.5, 0.0, 1.0, -3.0, // max at 1
        0.1, 0.2, 0.3, 0.4, 9.5, // max at 4
        7.0, 6.0, 5.0, 4.0, 3.0, // max at 0
    ];
    unsafe {
        std::ptr::copy_nonoverlapping(values.as_ptr(), ptr, values.len());
    }
    assert_eq!(logits.argmax_batch().unwrap(), vec![1, 4, 0]);
}

#[test]
fn spec_error_kinds_distinguish_input_vs_capacity() {
    let dir = TempDir::new("err-kinds");
    write_metadata(dir.path(), &base_metadata(512, "qwen3"));
    let spec = Spec::load(dir.path()).unwrap();

    assert_eq!(
        spec.check_tokens(&[]).unwrap_err().kind(),
        ErrorKind::InvalidInput
    );
    assert_eq!(
        spec.check_tokens(&vec![1; 513]).unwrap_err().kind(),
        ErrorKind::CapacityExceeded
    );
    assert_eq!(
        spec.check_request(&[1, 2], 0).unwrap_err().kind(),
        ErrorKind::InvalidInput
    );
    assert_eq!(
        spec.check_request(&[1, 2], 511).unwrap_err().kind(),
        ErrorKind::CapacityExceeded
    );
}

#[test]
fn copy_bytes_cpu_enforces_contiguity_and_bounds() {
    let mut base = Tensor::empty_contiguous(&[2, 4], DType::Float32, Device::Cpu).unwrap();
    // 2 * 4 * 4 = 32 bytes capacity: oversized write must fail cleanly.
    let too_large = vec![0u8; 36];
    let err = base.copy_bytes_cpu(&too_large).unwrap_err();
    assert!(err.to_string().contains("exceeds tensor byte capacity"));

    // Non-contiguous slice along dim 1 (strides [4, 1], sizes [2, 2]) must fail cleanly.
    let mut non_contig = base.slice_dim(1, 0, 2).unwrap();
    assert!(!non_contig.is_contiguous().unwrap());
    let err = non_contig.copy_bytes_cpu(&[0u8; 16]).unwrap_err();
    assert!(err.to_string().contains("contiguous"));

    // Exact-sized write on contiguous tensor succeeds.
    let exact = vec![0x2Au8; 32];
    base.copy_bytes_cpu(&exact).unwrap();
}

#[test]
#[ignore = "requires GPU and XINFER_PACKAGES"]
fn gpu_step_slot_and_reset_slot_match_batch_bitwise() {
    let dir = PathBuf::from(
        std::env::var("XINFER_PACKAGES").expect("Set XINFER_PACKAGES to a compiled bundle"),
    );
    let model = Model::open(&dir).unwrap();
    let mut session = model.open_session(Some(4)).unwrap();

    // Baseline: generate 4 tokens on slot 0 for two distinct prompts.
    let prompt_a = [42i64, 13, 99, 7];
    let prompt_b = [11i64, 22, 33];
    let expected_a = session.generate(&prompt_a, 4, None).unwrap();
    let expected_b = session.generate(&prompt_b, 4, None).unwrap();

    // Now test slot 2 directly via step_slot (non-zero storage_offset) + interior hole in step_slots_range.
    session.reset(0.0).unwrap();
    let logits_a0 = session.step_slot(0, &prompt_a).unwrap();
    let tok_a0 = logits_a0.argmax_batch().unwrap()[0];
    assert_eq!(tok_a0, expected_a[0]);

    // Prefill prompt_b directly into slot 2 (leaving slot 1 as an untouched hole!).
    let logits_b0 = session.step_slot(2, &prompt_b).unwrap();
    let tok_b0 = logits_b0.argmax_batch().unwrap()[0];
    assert_eq!(tok_b0, expected_b[0]);

    // Batched decode across slots 0..=2 with slot 1 as an empty hole.
    let mut got_a = vec![tok_a0];
    let mut got_b = vec![tok_b0];
    for _ in 1..4 {
        let last_a = [*got_a.last().unwrap()];
        let last_b = [*got_b.last().unwrap()];
        let logits = session
            .step_slots_range(0, &[&last_a, &[], &last_b])
            .unwrap();
        let next = logits.argmax_batch().unwrap();
        got_a.push(next[0]);
        got_b.push(next[2]);
    }
    assert_eq!(got_a, expected_a);
    assert_eq!(got_b, expected_b);

    // Recycle slot 2 with prompt_a (exercises reset_slot for DeltaNet + KV overwrite) and verify exact parity.
    session.reset_slot(2).unwrap();
    let mut recycled = Vec::new();
    let mut tok = session
        .step_slot(2, &prompt_a)
        .unwrap()
        .argmax_batch()
        .unwrap()[0];
    recycled.push(tok);
    for _ in 1..4 {
        tok = session
            .step_slot(2, &[tok])
            .unwrap()
            .argmax_batch()
            .unwrap()[0];
        recycled.push(tok);
    }
    assert_eq!(recycled, expected_a);
}
