"""Full-model package metadata shared by exporters and consumers."""

import copy
import math
from pathlib import Path
import re

from compiler.common.artifacts import read_json, require, sha256


METADATA_FILE = "metadata.json"
ENTRY_KINDS = ("prefill", "decode")
TOKENIZER_REQUIRED_FILES = ("tokenizer.json", "tokenizer_config.json")
TOKENIZER_OPTIONAL_FILES = ("chat_template.jinja",)
TOKENIZER_FILES = (*TOKENIZER_REQUIRED_FILES, *TOKENIZER_OPTIONAL_FILES)
FIELDS = (
    "model_type",  # Model family used to select runtime behavior.
    "torch_version",  # PyTorch version used for export and compilation.
    "cxx11_abi",  # PyTorch's _GLIBCXX_USE_CXX11_ABI setting.
    "cuda_compute_capability",  # Target GPU architecture as [major, minor].
    "cache_length",  # Allocated KV-cache capacity in tokens.
    "max_prefill_tokens",  # Maximum tokens accepted by one compiled prefill call.
    "vocab_size",  # Vocabulary size and exclusive token-ID upper bound.
    "eos_token_ids",  # Token IDs that stop greedy generation.
    "checkpoint",  # Prepared checkpoint manifest (model, revision and file facts).
    "state",  # Ordered contiguous in-place state shapes and dtypes.
    "entries",  # Compiled prefill/decode .pt2 sizes and SHA-256 checksums.
    "shared_weights",  # Shared weight archive identity, size and SHA-256 checksum.
    "tokenizer",  # Bundled tokenizer file sizes and SHA-256 checksums.
)
STATE_FIELDS = {"layer", "kind", "shape", "dtype"}
KV_STATE_KINDS = ("key", "value")
MAX_BATCH_SIZE = 8
MAX_PREFILL_TOKENS = 512
# Keep cumulative int32 KV lengths across all compiled batch slots below INT32_MAX.
MAX_CACHE_LENGTH = (2**31 - 1) // MAX_BATCH_SIZE
# Compiled artifact sizes/hashes may differ; shared weights are compared separately
# by logical identity and source mappings rather than archive checksum.
COMPARISON_EXCLUDED_FIELDS = {"entries", "shared_weights"}


def _check(condition, field):
    require(condition, f"Unsupported package metadata: {field}")


def _integer(value, minimum=0):
    return type(value) is int and minimum <= value <= 2**63 - 1


def _digest(value):
    return isinstance(value, str) and re.fullmatch("[0-9a-f]{64}", value) is not None


def is_kv_state(spec):
    """Return whether a state descriptor refers to a KV-cache tensor."""
    return spec["kind"] in KV_STATE_KINDS


def state_name(spec):
    """Derive a diagnostic label without storing a duplicate state identifier."""
    return f"layer_{spec['layer']}.{spec['kind']}"


def state_spec(layer, kind, shape, dtype):
    """Describe one contiguous caller-owned in-place state tensor."""
    shape = list(shape)
    return dict(
        layer=layer,
        kind=kind,
        shape=shape,
        dtype=dtype,
    )


def make_metadata(
    *,
    model_type,
    checkpoint,
    torch_version,
    cxx11_abi,
    cuda_compute_capability,
    cache_length,
    vocab_size,
    eos_token_ids,
    state,
    max_prefill_tokens=MAX_PREFILL_TOKENS,
):
    """Build the common schema; bundle publication supplies entry and archive facts."""
    eos = [eos_token_ids] if type(eos_token_ids) is int else list(eos_token_ids)
    states = copy.deepcopy(list(state))
    return dict(
        model_type=model_type,
        torch_version=str(torch_version),
        cxx11_abi=cxx11_abi,
        cuda_compute_capability=list(cuda_compute_capability),
        cache_length=cache_length,
        max_prefill_tokens=max_prefill_tokens,
        vocab_size=vocab_size,
        eos_token_ids=eos,
        checkpoint=copy.deepcopy(checkpoint),
        state=states,
    )


def validate_metadata(metadata, model_type=None):
    """Validate a complete bundle descriptor without loading Torch, files or CUDA."""
    _check(isinstance(metadata, dict), "expected an object")
    require(
        metadata.get("model_type") in ("qwen3", "qwen35", "qwen38")
        and (model_type is None or metadata["model_type"] == model_type),
        "Unsupported model type",
    )
    for field in FIELDS:
        _check(field in metadata, field)
    for field in metadata:
        _check(field in FIELDS, field)
    m = metadata
    _check(
        _integer(m["cache_length"], MAX_PREFILL_TOKENS) and m["cache_length"] <= MAX_CACHE_LENGTH,
        "cache_length",
    )
    _check(
        _integer(m["max_prefill_tokens"], 1)
        and m["max_prefill_tokens"] == MAX_PREFILL_TOKENS
        and m["max_prefill_tokens"] <= m["cache_length"],
        "max_prefill_tokens",
    )
    _check(_integer(m["vocab_size"], 1), "vocab_size")
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
        and all(
            isinstance(checkpoint.get(k), str) and checkpoint[k] for k in ("model", "revision")
        ),
        "checkpoint",
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
            and shape[0] == MAX_BATCH_SIZE
            and math.prod(shape) <= (2**63 - 1) // 4,
            "state",
        )
        _check(item["dtype"] == ("float32" if kind == "recurrent" else "bfloat16"), "state")
        if kind in KV_STATE_KINDS:
            _check(shape[2] == m["cache_length"], "state")
    entries = m["entries"]
    _check(
        isinstance(entries, dict)
        and set(entries) == set(ENTRY_KINDS)
        and all(
            isinstance(entries[kind], dict)
            and set(entries[kind]) == {"bytes", "sha256"}
            and _integer(entries[kind]["bytes"], 1)
            and _digest(entries[kind]["sha256"])
            for kind in ENTRY_KINDS
        ),
        "entries",
    )
    weights = m["shared_weights"]
    _check(
        isinstance(weights, dict)
        and set(weights) == {"identity", "bytes", "sha256"}
        and _integer(weights["bytes"], 1)
        and _digest(weights["sha256"])
        and _digest(weights["identity"]),
        "shared_weights",
    )
    tokenizer = m["tokenizer"]
    _check(
        isinstance(tokenizer, dict)
        and set(TOKENIZER_REQUIRED_FILES) <= set(tokenizer)
        and set(tokenizer) <= set(TOKENIZER_FILES)
        and all(
            isinstance(entry, dict)
            and set(entry) == {"bytes", "sha256"}
            and _integer(entry["bytes"], 1)
            and _digest(entry["sha256"])
            for entry in tokenizer.values()
        ),
        "tokenizer",
    )


def read_package_metadata(directory, model_type=None):
    result = read_json(Path(directory) / METADATA_FILE)
    validate_metadata(result, model_type)
    return result


def finalize_metadata(metadata, *, directory, shared_weights):
    """Attach publication facts for entries, shared archive and tokenizer files, then validate."""
    directory = Path(directory)
    entries = {}
    for kind in ENTRY_KINDS:
        package_path = directory / f"{kind}.pt2"
        entries[kind] = dict(bytes=package_path.stat().st_size, sha256=sha256(package_path))
    tokenizer = {}
    for name in TOKENIZER_FILES:
        tokenizer_path = directory / name
        if tokenizer_path.is_file():
            tokenizer[name] = dict(
                bytes=tokenizer_path.stat().st_size, sha256=sha256(tokenizer_path)
            )
        elif name in TOKENIZER_REQUIRED_FILES:
            raise ValueError(f"Tokenizer file is missing: {name}")
    result = dict(
        metadata,
        entries=entries,
        shared_weights=copy.deepcopy(shared_weights),
        tokenizer=tokenizer,
    )
    validate_metadata(result)
    return result
