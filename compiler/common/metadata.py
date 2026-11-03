"""Versioned full-model package metadata shared by exporters and consumers."""

import copy
import math
from pathlib import Path
import re

from compiler.common.artifacts import canonical_json, read_json, require, sha256
from compiler.common.attention import OPERATOR_CONTRACT


SCHEMA_VERSION = 1
KV_EXTENT = dict(device="cpu", dtype="uint8", shape="[valid_length + 1]", data_read=False)
FIELDS = (
    "schema_version",  # Version of this metadata format.
    "model_type",  # Model family used to select runtime behavior.
    "entry_point",  # Package role: prefill or decode.
    "token_range",  # Inclusive [minimum, maximum] tokens accepted per call.
    "torch_version",  # PyTorch version used for export and compilation.
    "torch_cuda_version",  # CUDA version that PyTorch was built against.
    "cxx11_abi",  # PyTorch's _GLIBCXX_USE_CXX11_ABI setting.
    "cuda_compute_capability",  # Target GPU architecture as [major, minor].
    "batch_size",  # Fixed inference batch size.
    "dtype",  # Model activation dtype; state entries specify their own dtypes.
    "cache_length",  # Allocated KV-cache capacity in tokens.
    "vocab_size",  # Vocabulary size and exclusive token-ID upper bound.
    "eos_token_ids",  # Token IDs that stop greedy generation.
    "checkpoint",  # Source model ID, revision and checkpoint manifest.
    "state",  # Ordered state descriptors: layout, dtype and update policy.
    "input_order",  # Flattened AOTI input names in call order.
    "output_order",  # AOTI output names in return order.
    "logits",  # Shape and dtype of the last-token logits.
    "input_mutation",  # Which inputs the graph may modify in place.
    "attention_backend",  # Attention implementation used by the compiled model.
    "custom_attention",  # External attention operator name and ABI contract.
    "cache_input",  # Cache-buffer representation expected by the graph.
    "kv_extent",  # Contract for the shape-only CPU valid-prefix input.
    "export_options",  # Graph export settings that affect the package contract.
    "tf32",  # Whether CUDA matmul and cuDNN may use TF32.
    "triton_enable_fp_fusion",  # Allow Triton floating-point fusion, such as FMA.
    "inductor_configs",  # Explicit Inductor compilation overrides for this entry.
    "package_bytes",  # Compiled .pt2 file size in bytes.
    "package_sha256",  # SHA-256 checksum of the compiled .pt2 file.
    "shared_weights",  # Archive format, filename, identity, size and checksum.
    "model_specific",  # Model-family extras, such as geometry or DeltaNet settings.
    "provenance",  # Exporter source hashes, dependency versions and capture settings.
    "diagnostics",  # Per-entry compilation timings and graph diagnostics.
)
ENTRY_FIELDS = {"entry_point", "token_range", "package_bytes", "package_sha256", "diagnostics"}
STATE_FIELDS = {"name", "layer", "kind", "shape", "stride", "dtype", "update"}
# Entry points are paired by kind; artifact facts and provenance may differ.
# Shared weights are compared separately by logical identity and source mapping.
COMPARISON_EXCLUDED_FIELDS = {
    "entry_point", "package_bytes", "package_sha256", "shared_weights", "provenance", "diagnostics",
}


def _check(condition, field):
    require(condition, f"Unsupported package metadata: {field}")


def _integer(value, minimum=0):
    return type(value) is int and minimum <= value <= 2**63 - 1


def _digest(value):
    return isinstance(value, str) and re.fullmatch("[0-9a-f]{64}", value) is not None


def _same_json(left, right):
    return canonical_json(left) == canonical_json(right)


def contiguous_stride(shape):
    result, step = [], 1
    for size in reversed(shape):
        result.append(step)
        step *= size
    return list(reversed(result))


def state_spec(layer, kind, shape, dtype, update):
    """Describe one contiguous caller-owned input and its update contract."""
    shape = list(shape)
    return dict(name=f"layer_{layer}.{kind}", layer=layer, kind=kind, shape=shape,
                stride=contiguous_stride(shape), dtype=dtype, update=update)


def make_metadata(*, model_type, entry_point, token_range, checkpoint, torch_version,
                  torch_cuda_version, cxx11_abi, cuda_compute_capability, cache_length,
                  vocab_size, eos_token_ids, state, model_specific, return_state=False,
                  provenance=None, diagnostics=None, triton_enable_fp_fusion=True):
    """Build the common schema; bundle publication supplies file/archive fields."""
    eos = [eos_token_ids] if type(eos_token_ids) is int else list(eos_token_ids)
    states = copy.deepcopy(list(state))
    names = [item["name"] for item in states]
    return dict(
        schema_version=SCHEMA_VERSION, model_type=model_type, entry_point=entry_point,
        token_range=list(token_range), torch_version=str(torch_version),
        torch_cuda_version=torch_cuda_version, cxx11_abi=cxx11_abi,
        cuda_compute_capability=list(cuda_compute_capability), batch_size=1,
        dtype="bfloat16", cache_length=cache_length, vocab_size=vocab_size,
        eos_token_ids=eos, checkpoint=copy.deepcopy(checkpoint), state=states,
        input_order=["input_ids", "cache_position", *names, "kv_extent"],
        output_order=["last_logits", *(names if return_state else [])],
        logits=dict(shape=[1, vocab_size], dtype="bfloat16"), input_mutation="kv_only",
        attention_backend="flash", custom_attention=copy.deepcopy(OPERATOR_CONTRACT),
        cache_input="full_capacity_with_extent", kv_extent=dict(KV_EXTENT),
        export_options=dict(backed_size_oblivious=False), tf32=False,
        triton_enable_fp_fusion=triton_enable_fp_fusion,
        model_specific=copy.deepcopy(model_specific), provenance=copy.deepcopy(provenance or {}),
        diagnostics=copy.deepcopy(diagnostics or {}),
    )


def _schema(metadata):
    require(isinstance(metadata, dict) and type(metadata.get("schema_version")) is int and
            metadata["schema_version"] == SCHEMA_VERSION,
            "Unsupported package metadata schema version; convert retained metadata or re-export both packages")


def validate_metadata(metadata, model_type=None, entry_point=None):
    """Validate a complete v1 descriptor without loading Torch, files or CUDA."""
    _schema(metadata)
    require(metadata.get("model_type") in ("qwen3", "qwen35") and
            (model_type is None or metadata["model_type"] == model_type), "Unsupported model type")
    for field in FIELDS:
        _check(field in metadata, field)
    for field in metadata:
        _check(field in FIELDS, field)
    m = metadata
    _check(m["entry_point"] in ("prefill", "decode") and
           (entry_point is None or m["entry_point"] == entry_point), "entry_point")
    _check(type(m["batch_size"]) is int and m["batch_size"] == 1, "batch_size")
    _check(m["dtype"] == "bfloat16", "dtype")
    _check(type(m["cache_length"]) is int and m["cache_length"] == 1024, "cache_length")
    _check(_integer(m["vocab_size"], 1), "vocab_size")
    tokens = m["token_range"]
    _check(isinstance(tokens, list) and len(tokens) == 2 and
           all(_integer(n, 1) for n in tokens) and
           tokens == ([1, 1] if m["entry_point"] == "decode" else [2, 512]), "token_range")
    for field in ("torch_version", "torch_cuda_version"):
        _check(isinstance(m[field], str) and bool(m[field]), field)
    _check(type(m["cxx11_abi"]) is bool, "cxx11_abi")
    sm = m["cuda_compute_capability"]
    _check(isinstance(sm, list) and len(sm) == 2 and
           _integer(sm[0], 1) and _integer(sm[1]), "cuda_compute_capability")
    eos = m["eos_token_ids"]
    _check(isinstance(eos, list) and bool(eos) and
           all(_integer(token) and token < m["vocab_size"] for token in eos), "eos_token_ids")
    checkpoint = m["checkpoint"]
    _check(isinstance(checkpoint, dict) and set(checkpoint) == {"model_id", "revision", "manifest"} and
           all(isinstance(checkpoint.get(k), str) and checkpoint[k] for k in ("model_id", "revision")) and
           isinstance(checkpoint["manifest"], dict), "checkpoint")
    _check(m["attention_backend"] == "flash", "attention_backend")
    _check(_same_json(m["custom_attention"], OPERATOR_CONTRACT), "custom_attention")
    _check(m["cache_input"] == "full_capacity_with_extent", "cache_input")
    _check(m["input_mutation"] == "kv_only", "input_mutation")
    _check(_same_json(m["kv_extent"], KV_EXTENT), "kv_extent")
    _check(_same_json(m["export_options"], {"backed_size_oblivious": False}), "export_options")
    _check(m["tf32"] is False, "tf32")
    _check(type(m["triton_enable_fp_fusion"]) is bool, "triton_enable_fp_fusion")
    _check(isinstance(m["inductor_configs"], dict), "inductor_configs")
    states = m["state"]
    _check(isinstance(states, list) and bool(states), "state")
    names = []
    for item in states:
        _check(isinstance(item, dict) and set(item) == STATE_FIELDS, "state")
        kind = item["kind"]
        _check(kind in ("key", "value", "conv", "recurrent") and _integer(item["layer"]), "state")
        _check(item["name"] == f"layer_{item['layer']}.{kind}" and item["name"] not in names, "state")
        names.append(item["name"])
        shape = item["shape"]
        _check(isinstance(shape, list) and len(shape) == (3 if kind == "conv" else 4) and
               all(_integer(n, 1) for n in shape) and shape[0] == 1 and
               math.prod(shape) <= (2**63 - 1) // 4, "state")
        stride = item["stride"]
        _check(isinstance(stride, list) and all(_integer(n, 1) for n in stride) and
               stride == contiguous_stride(shape), "state")
        _check(item["dtype"] == ("float32" if kind == "recurrent" else "bfloat16"), "state")
        _check(item["update"] == ("inplace" if kind in ("key", "value") else "replace"), "state")
        if kind in ("key", "value"):
            _check(shape[2] == m["cache_length"], "state")
    _check(m["input_order"] == ["input_ids", "cache_position", *names, "kv_extent"], "input_order")
    _check(m["output_order"] in (["last_logits"], ["last_logits", *names]), "output_order")
    _check(_same_json(m["logits"], {"shape": [1, m["vocab_size"]], "dtype": "bfloat16"}), "logits")
    _check(_integer(m["package_bytes"], 1), "package_bytes")
    _check(_digest(m["package_sha256"]), "package_sha256")
    weights = m["shared_weights"]
    _check(isinstance(weights, dict) and set(weights) == {"format", "file", "identity", "bytes", "sha256"} and
           type(weights["format"]) is int and weights["format"] == 2 and weights["file"] == "weights.pt" and
           _integer(weights["bytes"], 1) and _digest(weights["sha256"]) and _digest(weights["identity"]), "shared_weights")
    for field in ("model_specific", "provenance", "diagnostics"):
        _check(isinstance(m[field], dict), field)


def validate_metadata_pair(prefill, decode, model_type=None):
    """Reject mixed schema/model/contracts before file, archive or device access."""
    for m in (prefill, decode):
        _schema(m)
    require(prefill.get("model_type") == decode.get("model_type"), "Package model types differ")
    for m in (prefill, decode):
        require(m.get("model_type") in ("qwen3", "qwen35") and
                (model_type is None or m["model_type"] == model_type), "Unsupported model type")
    for field in FIELDS:
        if field not in ENTRY_FIELDS:
            require((field not in prefill and field not in decode) or
                    (field in prefill and field in decode and _same_json(prefill[field], decode[field])),
                    f"Package metadata differs: {field}")
    validate_metadata(prefill, model_type, "prefill")
    validate_metadata(decode, model_type, "decode")


def read_package_metadata(directory, model_type=None):
    directory = Path(directory)
    result = {kind: read_json(directory / f"{kind}.json") for kind in ("prefill", "decode")}
    validate_metadata_pair(result["prefill"], result["decode"], model_type)
    return result


def finalize_metadata(metadata, *, package_path, inductor_configs, shared_weights):
    """Attach publication facts to one common descriptor and validate it."""
    package_path = Path(package_path)
    result = dict(metadata, package_bytes=package_path.stat().st_size,
                  package_sha256=sha256(package_path), inductor_configs=copy.deepcopy(inductor_configs),
                  shared_weights=copy.deepcopy(shared_weights))
    validate_metadata(result)
    return result
