"""Versioned full-model package metadata shared by exporters and consumers."""

import copy
import math
from pathlib import Path
import re

from compiler.common.artifacts import canonical_json, read_json, require, sha256


SCHEMA_VERSION = 3
RUNTIME_ABI_VERSION = 1
FIELDS = (
    "schema_version",  # Version of this metadata format.
    "runtime_abi_version",  # Compiled calling convention, state ownership and attention ABI.
    "model_type",  # Model family used to select runtime behavior.
    "entry_point",  # Package role: prefill or decode.
    "token_range",  # Inclusive [minimum, maximum] tokens accepted per call.
    "torch_version",  # PyTorch version used for export and compilation.
    "cxx11_abi",  # PyTorch's _GLIBCXX_USE_CXX11_ABI setting.
    "cuda_compute_capability",  # Target GPU architecture as [major, minor].
    "cache_length",  # Allocated KV-cache capacity in tokens.
    "vocab_size",  # Vocabulary size and exclusive token-ID upper bound.
    "eos_token_ids",  # Token IDs that stop greedy generation.
    "checkpoint",  # Source model ID, revision and checkpoint manifest.
    "state",  # Ordered contiguous state shapes, dtypes and update policies.
    "export_options",  # Graph export settings that affect the package contract.
    "tf32",  # Whether CUDA matmul and cuDNN may use TF32.
    "triton_enable_fp_fusion",  # Allow Triton floating-point fusion, such as FMA.
    "inductor_configs",  # Precision and external-constant compilation contract.
    "package_bytes",  # Compiled .pt2 file size in bytes.
    "package_sha256",  # SHA-256 checksum of the compiled .pt2 file.
    "shared_weights",  # Archive format, filename, identity, size and checksum.
    "model_specific",  # Model-specific numerical and validation settings.
)
ENTRY_FIELDS = {"entry_point", "token_range", "package_bytes", "package_sha256"}
STATE_FIELDS = {"layer", "kind", "shape", "dtype", "update"}
INDUCTOR_FIELDS = {
    "emulate_precision_casts",
    "aot_inductor.package_constants_in_so",
    "always_keep_tensor_constants",
}
# Entry points are paired by kind; compiled artifact facts may differ.
# Shared weights are compared separately by logical identity and source mapping.
COMPARISON_EXCLUDED_FIELDS = {
    "entry_point",
    "package_bytes",
    "package_sha256",
    "shared_weights",
}


def _check(condition, field):
    require(condition, f"Unsupported package metadata: {field}")


def _integer(value, minimum=0):
    return type(value) is int and minimum <= value <= 2**63 - 1


def _digest(value):
    return isinstance(value, str) and re.fullmatch("[0-9a-f]{64}", value) is not None


def _same_json(left, right):
    return canonical_json(left) == canonical_json(right)


def state_name(spec):
    """Derive a diagnostic label without storing a duplicate state identifier."""
    return f"layer_{spec['layer']}.{spec['kind']}"


def state_spec(layer, kind, shape, dtype, update):
    """Describe one contiguous caller-owned input and its update contract."""
    shape = list(shape)
    return dict(
        layer=layer,
        kind=kind,
        shape=shape,
        dtype=dtype,
        update=update,
    )


def make_metadata(
    *,
    model_type,
    entry_point,
    token_range,
    checkpoint,
    torch_version,
    cxx11_abi,
    cuda_compute_capability,
    cache_length,
    vocab_size,
    eos_token_ids,
    state,
    model_specific,
    triton_enable_fp_fusion=True,
):
    """Build the common schema; bundle publication supplies file/archive fields."""
    eos = [eos_token_ids] if type(eos_token_ids) is int else list(eos_token_ids)
    states = copy.deepcopy(list(state))
    return dict(
        schema_version=SCHEMA_VERSION,
        runtime_abi_version=RUNTIME_ABI_VERSION,
        model_type=model_type,
        entry_point=entry_point,
        token_range=list(token_range),
        torch_version=str(torch_version),
        cxx11_abi=cxx11_abi,
        cuda_compute_capability=list(cuda_compute_capability),
        cache_length=cache_length,
        vocab_size=vocab_size,
        eos_token_ids=eos,
        checkpoint=copy.deepcopy(checkpoint),
        state=states,
        export_options=dict(backed_size_oblivious=False),
        tf32=False,
        triton_enable_fp_fusion=triton_enable_fp_fusion,
        model_specific=copy.deepcopy(model_specific),
    )


def _schema(metadata):
    require(
        isinstance(metadata, dict)
        and type(metadata.get("schema_version")) is int
        and metadata["schema_version"] == SCHEMA_VERSION,
        "Unsupported package metadata schema version; convert retained metadata or re-export both packages",
    )


def _runtime_abi(metadata):
    require(
        type(metadata.get("runtime_abi_version")) is int
        and metadata["runtime_abi_version"] == RUNTIME_ABI_VERSION,
        "Unsupported runtime ABI version; convert retained metadata or re-export both packages",
    )


def validate_metadata(metadata, model_type=None, entry_point=None):
    """Validate a complete v3 descriptor without loading Torch, files or CUDA."""
    _schema(metadata)
    _runtime_abi(metadata)
    require(
        metadata.get("model_type") in ("qwen3", "qwen35")
        and (model_type is None or metadata["model_type"] == model_type),
        "Unsupported model type",
    )
    for field in FIELDS:
        _check(field in metadata, field)
    for field in metadata:
        _check(field in FIELDS, field)
    m = metadata
    _check(
        m["entry_point"] in ("prefill", "decode")
        and (entry_point is None or m["entry_point"] == entry_point),
        "entry_point",
    )
    _check(type(m["cache_length"]) is int and m["cache_length"] == 1024, "cache_length")
    _check(_integer(m["vocab_size"], 1), "vocab_size")
    tokens = m["token_range"]
    _check(
        isinstance(tokens, list)
        and len(tokens) == 2
        and all(_integer(n, 1) for n in tokens)
        and tokens == ([1, 1] if m["entry_point"] == "decode" else [2, 512]),
        "token_range",
    )
    _check(isinstance(m["torch_version"], str) and bool(m["torch_version"]), "torch_version")
    _check(type(m["cxx11_abi"]) is bool, "cxx11_abi")
    sm = m["cuda_compute_capability"]
    _check(
        isinstance(sm, list) and len(sm) == 2 and _integer(sm[0], 1) and _integer(sm[1]),
        "cuda_compute_capability",
    )
    eos = m["eos_token_ids"]
    _check(
        isinstance(eos, list)
        and bool(eos)
        and all(_integer(token) and token < m["vocab_size"] for token in eos),
        "eos_token_ids",
    )
    checkpoint = m["checkpoint"]
    _check(
        isinstance(checkpoint, dict)
        and set(checkpoint) == {"model_id", "revision", "manifest"}
        and all(
            isinstance(checkpoint.get(k), str) and checkpoint[k] for k in ("model_id", "revision")
        )
        and isinstance(checkpoint["manifest"], dict),
        "checkpoint",
    )
    _check(_same_json(m["export_options"], {"backed_size_oblivious": False}), "export_options")
    _check(m["tf32"] is False, "tf32")
    _check(type(m["triton_enable_fp_fusion"]) is bool, "triton_enable_fp_fusion")
    options = m["inductor_configs"]
    _check(
        isinstance(options, dict)
        and set(options) == INDUCTOR_FIELDS
        and options["emulate_precision_casts"] is True
        and options["aot_inductor.package_constants_in_so"] is False
        and options["always_keep_tensor_constants"] is True,
        "inductor_configs",
    )
    states = m["state"]
    _check(isinstance(states, list) and bool(states), "state")
    names = []
    for item in states:
        _check(isinstance(item, dict) and set(item) == STATE_FIELDS, "state")
        kind = item["kind"]
        _check(kind in ("key", "value", "conv", "recurrent") and _integer(item["layer"]), "state")
        name = state_name(item)
        _check(name not in names, "state")
        names.append(name)
        shape = item["shape"]
        _check(
            isinstance(shape, list)
            and len(shape) == (3 if kind == "conv" else 4)
            and all(_integer(n, 1) for n in shape)
            and shape[0] == 1
            and math.prod(shape) <= (2**63 - 1) // 4,
            "state",
        )
        _check(item["dtype"] == ("float32" if kind == "recurrent" else "bfloat16"), "state")
        _check(item["update"] == ("inplace" if kind in ("key", "value") else "replace"), "state")
        if kind in ("key", "value"):
            _check(shape[2] == m["cache_length"], "state")
    _check(_integer(m["package_bytes"], 1), "package_bytes")
    _check(_digest(m["package_sha256"]), "package_sha256")
    weights = m["shared_weights"]
    _check(
        isinstance(weights, dict)
        and set(weights) == {"format", "file", "identity", "bytes", "sha256"}
        and type(weights["format"]) is int
        and weights["format"] == 2
        and weights["file"] == "weights.pt"
        and _integer(weights["bytes"], 1)
        and _digest(weights["sha256"])
        and _digest(weights["identity"]),
        "shared_weights",
    )
    _check(isinstance(m["model_specific"], dict), "model_specific")


def validate_metadata_pair(prefill, decode, model_type=None):
    """Reject mixed schema/model/contracts before file, archive or device access."""
    for m in (prefill, decode):
        _schema(m)
        _runtime_abi(m)
    require(prefill.get("model_type") == decode.get("model_type"), "Package model types differ")
    for m in (prefill, decode):
        require(
            m.get("model_type") in ("qwen3", "qwen35")
            and (model_type is None or m["model_type"] == model_type),
            "Unsupported model type",
        )
    for field in FIELDS:
        if field not in ENTRY_FIELDS:
            require(
                (field not in prefill and field not in decode)
                or (
                    field in prefill
                    and field in decode
                    and _same_json(prefill[field], decode[field])
                ),
                f"Package metadata differs: {field}",
            )
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
    result = dict(
        metadata,
        package_bytes=package_path.stat().st_size,
        package_sha256=sha256(package_path),
        inductor_configs={
            key: copy.deepcopy(value)
            for key, value in inductor_configs.items()
            if key in INDUCTOR_FIELDS
        },
        shared_weights=copy.deepcopy(shared_weights),
    )
    validate_metadata(result)
    return result
