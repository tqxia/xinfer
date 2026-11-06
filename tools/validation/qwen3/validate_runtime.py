"""Validate the native Flash CLI, deployment contract and exact Python AOTI parity."""

import argparse
import json
import os
from pathlib import Path
import subprocess

import numpy as np
import torch

from compiler.common.attention import cache_extent, load_attention_ops
from compiler.common.weights import SharedWeights, validate_shared_weights
from compiler.common.artifacts import (
    MODEL_MANIFEST_NAME,
    default_build_dir,
    read_json,
    require,
    sha256,
)
from compiler.common.metadata import (
    RUNTIME_ABI_VERSION,
    SCHEMA_VERSION,
    read_package_metadata,
    validate_metadata,
)
from compiler.qwen3.packages import validate_model_metadata
from compiler.qwen3.config import DEFAULT_MODEL
from tools.common import native_attention_library
from tools.prepare.common import encode_prompt, load_tokenizer, write_ids
from tools.validation.common.prompts import chat_prompts
from tools.validation.common.validation import read_ids as ids_from


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--native", type=Path, default=default_build_dir("qwen3") / "xinfer_aoti")
parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL)
parser.add_argument("--packages", type=Path, default=Path("artifacts/aoti"))
parser.add_argument("--output", type=Path, default=Path("artifacts/native-smoke"))
args = parser.parse_args()
binary, packages, out = args.native.resolve(), args.packages, args.output
if out.exists():
    parser.error("Output directory already exists; choose a fresh --output")
out.mkdir(parents=True)
env = {
    key: value
    for key, value in os.environ.items()
    if key not in ("VIRTUAL_ENV", "PYTHONPATH", "PYTHONHOME", "LD_LIBRARY_PATH")
}
env.update(PATH="/usr/bin:/bin", PYTHONHOME="/nonexistent-python-home")
report = dict(fixtures=[], rejections=[], boundaries=[], native_processes=[])
linked = subprocess.check_output(["ldd", str(binary)], text=True)
(out / "ldd.txt").write_text(linked)
assert "not found" not in linked
assert "libpython" not in linked and "libtorch_python" not in linked
metadata = read_package_metadata(packages, model_type="qwen3")
for entry in metadata.values():
    validate_model_metadata(entry)
spec = metadata["decode"]
validate_shared_weights(packages, metadata["prefill"], spec)
attention_library = native_attention_library(binary)
require(
    attention_library is not None,
    "Qwen3 runtime is missing its required native attention library; rebuild the runtime",
)
os.environ["XINFER_ATTENTION_OPS_LIBRARY"] = str(attention_library)
load_attention_ops(attention_library, model="qwen3")
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
native_traces = []


def run_native(arguments, name, environment=env, timeout=120):
    result = subprocess.run(
        [str(binary), *map(str, arguments)],
        env=environment,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    (out / f"{name}.log").write_text(result.stderr)
    (out / f"{name}.stdout.log").write_text(result.stdout)
    report["native_processes"].append(dict(name=name, returncode=result.returncode))
    return result


def failure_detail(result, name):
    return dict(case=name, returncode=result.returncode, stdout=result.stdout, stderr=result.stderr)


@torch.inference_mode()
def compare_native_trace(prompt, generated, trace_path, budget):
    assert 1 <= len(generated) <= budget
    assert not any(token in spec["eos_token_ids"] for token in generated[:-1])
    assert len(generated) == budget or generated[-1] in spec["eos_token_ids"]
    rows = np.fromfile(trace_path, dtype="<f4").reshape(len(generated), spec["vocab_size"])
    assert np.isfinite(rows).all()
    cache = tuple(
        torch.zeros(state["shape"], dtype=getattr(torch, state["dtype"]), device="cuda")
        for state in spec["state"]
    )
    tokens, position = prompt, 0
    for token, row in zip(generated, rows):
        ids = torch.tensor([tokens], dtype=torch.int64, device="cuda")
        positions = torch.arange(position, position + len(tokens), device="cuda")
        scores = (decode if len(tokens) == 1 else prefill)(
            ids, positions, cache, cache_extent(position + len(tokens))
        )
        assert torch.isfinite(scores).all().item()
        assert scores.argmax(-1).item() == token
        assert scores.float().cpu().numpy()[0].tobytes() == row.tobytes()
        position += len(tokens)
        tokens = [token]
    assert all(torch.isfinite(tensor).all().item() for tensor in cache)


require(
    read_json(args.model_dir / MODEL_MANIFEST_NAME) == spec["checkpoint"]["manifest"],
    "Checkpoint manifest differs",
)
tokenizer = load_tokenizer(args.model_dir)
prompt_directory = out / "prompts"
prompt_directory.mkdir()
prompt_paths = {}
for case in sorted(chat_prompts("qwen3"), key=lambda item: item["name"]):
    name = case["name"]
    prompt = encode_prompt(tokenizer, case["text"])
    prompt_path = prompt_directory / f"{name}.ids"
    write_ids(prompt_path, prompt)
    prompt_paths[name] = prompt_path
    run_env = env | ({"LD_DEBUG": "libs"} if name == "capital" else {})
    ids, trace = out / f"{name}.ids", out / f"{name}.f32"
    budget = 64
    result = run_native(
        [packages, prompt_path, ids, "--max-new-tokens", budget, "--trace", trace],
        name,
        environment=run_env,
    )
    assert result.returncode == 0, failure_detail(result, name)
    actual = ids_from(ids)
    if name == "capital":
        assert "libpython" not in result.stderr and "libtorch_python" not in result.stderr
    entry = dict(name=name, tokens=len(actual), generated_ids=actual)
    report["fixtures"].append(entry)
    native_traces.append((entry, prompt, actual, trace, budget))
    print(report["fixtures"][-1], flush=True)

capital_prompt = prompt_paths["capital"]

bad = out / "invalid.ids"
cases = [
    ("empty", "", [], "Expected 1..512 input tokens"),
    ("negative", "-1", [], "Token ID out of range"),
    ("vocab", "151936", [], "Token ID out of range"),
    ("fraction", "1.5", [], "Invalid integer"),
    ("overflow", "999999999999999999999999999999", [], "Invalid integer"),
    ("long_prompt", "1 " * 513, [], "Expected 1..512 input tokens"),
    ("zero_budget", "1", ["--max-new-tokens", "0"], "Generation budget"),
    ("negative_budget", "1", ["--max-new-tokens", "-1"], "Generation budget"),
    ("context_budget", "1 " * 512, ["--max-new-tokens", "513"], "Generation budget"),
    ("unknown_flag", "1", ["--unknown", "1"], "Unknown option"),
    ("missing_value", "1", ["--trace"], "Missing option value"),
    (
        "trace_collision",
        "1",
        ["--trace", str(out / "invalid-output.ids")],
        "Trace path must differ",
    ),
]
for name, content, extra, message in cases:
    bad.write_text(content)
    result = run_native(
        [packages, bad, out / "invalid-output.ids", *extra],
        name,
        environment=env | {"CUDA_VISIBLE_DEVICES": ""},
        timeout=20,
    )
    assert result.returncode != 0 and message in result.stderr, failure_detail(result, name)
    report["rejections"].append(name)

# Both models use the same no-overwrite policy.
existing = out / "existing-output.ids"
existing.write_text("keep existing output\n")
result = run_native(
    [packages, capital_prompt, existing],
    "existing_output",
    environment=env | {"CUDA_VISIBLE_DEVICES": ""},
    timeout=20,
)
assert result.returncode != 0 and "Output already exists" in result.stderr, failure_detail(
    result, "existing_output"
)
assert existing.read_text() == "keep existing output\n"
report["rejections"].append("existing_output")

# All invalid package contracts must fail before CUDA or weight loading.
mixed = out / "invalid-packages"
mixed.mkdir()
package_cases = (
    ("mixed_numerical_settings", "Package metadata differs: triton_enable_fp_fusion"),
    ("legacy_eager_numerics", "Unsupported package metadata: eager_numerics"),
    ("missing_runtime_abi", "Unsupported runtime ABI version"),
    ("unknown_runtime_abi", "Unsupported runtime ABI version"),
    ("mixed_runtime_abi", "Unsupported runtime ABI version"),
    ("boolean_runtime_abi", "Unsupported runtime ABI version"),
    ("float_runtime_abi", "Unsupported runtime ABI version"),
    ("mixed_state_layout", "Package metadata differs: state"),
    ("odd_state_count", "Unsupported package metadata: state"),
    ("invalid_cache_shape", "Unsupported Flash cache shape"),
    ("mixed_export_settings", "Package metadata differs: export_options"),
)
package_cases += tuple(
    ("legacy_field_" + field, "Unsupported package metadata: " + field)
    for field in (
        "batch_size",
        "dtype",
        "input_mutation",
        "attention_backend",
        "custom_attention",
        "cache_input",
        "kv_extent",
    )
)
for name, message in package_cases:
    for kind in ("prefill", "decode"):
        metadata = json.loads((packages / (kind + ".json")).read_text())
        if name == "mixed_numerical_settings" and kind == "prefill":
            metadata["triton_enable_fp_fusion"] = not metadata["triton_enable_fp_fusion"]
        elif name == "legacy_eager_numerics":
            metadata["eager_numerics"] = True
        elif name.startswith("legacy_field_"):
            metadata[name.removeprefix("legacy_field_")] = False
        elif name == "missing_runtime_abi":
            del metadata["runtime_abi_version"]
        elif name == "unknown_runtime_abi" or (name == "mixed_runtime_abi" and kind == "prefill"):
            metadata["runtime_abi_version"] = RUNTIME_ABI_VERSION + 1
        elif name == "boolean_runtime_abi":
            metadata["runtime_abi_version"] = True
        elif name == "float_runtime_abi":
            metadata["runtime_abi_version"] = float(RUNTIME_ABI_VERSION)
        elif name == "mixed_state_layout" and kind == "prefill":
            metadata["state"][0]["shape"][-1] += 1
        elif name == "odd_state_count":
            metadata["state"].pop()
        elif name == "invalid_cache_shape":
            metadata["state"][0]["shape"][1] = 4
        elif name == "mixed_export_settings" and kind == "prefill":
            metadata["export_options"]["backed_size_oblivious"] = True
        (mixed / (kind + ".json")).write_text(json.dumps(metadata))
        if (
            name.startswith("legacy_")
            or type(metadata.get("runtime_abi_version")) is not int
            or metadata["runtime_abi_version"] != RUNTIME_ABI_VERSION
        ):
            try:
                validate_metadata(metadata, model_type="qwen3")
            except ValueError as error:
                assert message in str(error), (name, str(error))
            else:
                raise AssertionError(f"Python accepted invalid package: {name}")
    result = run_native(
        [mixed, capital_prompt, out / "mixed.ids"],
        name,
        environment=env | {"CUDA_VISIBLE_DEVICES": ""},
        timeout=20,
    )
    assert result.returncode != 0 and message in result.stderr, failure_detail(result, name)
    report["rejections"].append(name)


def package_fixture(name, metadata, with_weights=False):
    directory = out / name
    directory.mkdir()
    for kind, item in metadata.items():
        (directory / f"{kind}.json").write_text(json.dumps(item))
        (directory / f"{kind}.pt2").symlink_to((packages / f"{kind}.pt2").resolve())
    if with_weights:
        (directory / "weights.pt").symlink_to((packages / "weights.pt").resolve())
    return directory


def metadata_with_sizes():
    return {
        kind: dict(
            json.loads((packages / f"{kind}.json").read_text()),
            package_bytes=(packages / f"{kind}.pt2").stat().st_size,
        )
        for kind in ("prefill", "decode")
    }


# Dispatch failures must not require CUDA or an archive.
for name, version in (
    ("missing_schema_version", None),
    ("unknown_schema_version", 99),
    ("mixed_schema_version", SCHEMA_VERSION + 1),
):
    directory = out / name
    directory.mkdir()
    for kind in ("prefill", "decode"):
        item = json.loads((packages / f"{kind}.json").read_text())
        if version is None:
            item.pop("schema_version")
        elif name != "mixed_schema_version" or kind == "decode":
            item["schema_version"] = version
        (directory / f"{kind}.json").write_text(json.dumps(item))
        if version is None or item["schema_version"] != SCHEMA_VERSION:
            try:
                validate_metadata(item, model_type="qwen3")
            except ValueError as error:
                assert "Unsupported package metadata schema version" in str(error)
            else:
                raise AssertionError(f"Python accepted invalid metadata version: {name}")
    output = directory / "generated.ids"
    result = run_native(
        [directory, capital_prompt, output],
        name,
        environment=env | {"CUDA_VISIBLE_DEVICES": ""},
        timeout=20,
    )
    assert (
        result.returncode != 0 and "Unsupported package metadata schema version" in result.stderr
    ), failure_detail(result, name)
    assert not output.exists()
    report["rejections"].append(name)


for name, tags, message in (
    ("mixed_model_types", ("qwen3", "qwen35"), "Package model types differ"),
    ("missing_prefill_model_type", (None, "qwen3"), "Package model types differ"),
    ("missing_decode_model_type", ("qwen3", None), "Package model types differ"),
    ("unknown_model_type", ("unknown", "unknown"), "Unsupported model type"),
):
    directory = out / name
    directory.mkdir()
    for kind, tag in zip(("prefill", "decode"), tags):
        metadata = json.loads((packages / f"{kind}.json").read_text())
        metadata.pop("model_type", None)
        if tag is not None:
            metadata["model_type"] = tag
        (directory / f"{kind}.json").write_text(json.dumps(metadata))
    output = directory / "generated.ids"
    result = run_native(
        [directory, capital_prompt, output],
        name,
        environment=env | {"CUDA_VISIBLE_DEVICES": ""},
        timeout=20,
    )
    assert result.returncode != 0 and message in result.stderr, failure_detail(result, name)
    assert not output.exists()
    report["rejections"].append(name)

# Declared sizes must fail before archive validation, even with CUDA hidden.
for name, kind, missing in (
    ("wrong_prefill_size", "prefill", False),
    ("wrong_decode_size", "decode", False),
    ("missing_prefill_size", "prefill", True),
    ("missing_decode_size", "decode", True),
):
    metadata = metadata_with_sizes()
    if missing:
        del metadata[kind]["package_bytes"]
    else:
        metadata[kind]["package_bytes"] += 1
    directory = package_fixture(name, metadata)
    output = directory / "generated.ids"
    message = (
        "Unsupported package metadata: package_bytes" if missing else "Package file size mismatch"
    )
    result = run_native(
        [directory, capital_prompt, output],
        name,
        environment=env | {"CUDA_VISIBLE_DEVICES": ""},
        timeout=20,
    )
    assert result.returncode != 0 and message in result.stderr, failure_detail(result, name)
    assert not output.exists(), name
    report["rejections"].append(name)

# Each package checksum is required and checked independently of the weight archive.
for name, kind, missing in (
    ("wrong_prefill_hash", "prefill", False),
    ("wrong_decode_hash", "decode", False),
    ("missing_prefill_hash", "prefill", True),
    ("missing_decode_hash", "decode", True),
):
    metadata = metadata_with_sizes()
    if missing:
        del metadata[kind]["package_sha256"]
    else:
        metadata[kind]["package_sha256"] = "0" * 64
    directory = package_fixture(name, metadata)
    output = directory / "generated.ids"
    message = "Unsupported package metadata: package_sha256" if missing else "checksum mismatch"
    result = run_native(
        [directory, capital_prompt, output],
        name,
        environment=env | {"CUDA_VISIBLE_DEVICES": ""},
        timeout=20,
    )
    assert result.returncode != 0 and message in result.stderr, failure_detail(result, name)
    assert not output.exists(), name
    report["rejections"].append(name)

# Valid size metadata must still pass complete loading and native/Python parity.
directory = package_fixture("declared_package_sizes", metadata_with_sizes(), with_weights=True)
prompt_path = capital_prompt
ids_path, trace_path = directory / "generated.ids", directory / "logits.f32"
result = run_native(
    [directory, prompt_path, ids_path, "--max-new-tokens", "1", "--trace", trace_path],
    "declared_package_sizes",
)
assert result.returncode == 0, failure_detail(result, "declared_package_sizes")
generated = ids_from(ids_path)
entry = dict(contract="declared_package_sizes", tokens=len(generated))
report.setdefault("package_controls", []).append(entry)
native_traces.append((entry, ids_from(prompt_path), generated, trace_path, 1))

# The complete, checksum-valid bundle must reject an incompatible actual GPU.
metadata = metadata_with_sizes()
for item in metadata.values():
    item["cuda_compute_capability"] = [99, 0]
directory = package_fixture("gpu_architecture", metadata, with_weights=True)
output = directory / "generated.ids"
result = run_native([directory, prompt_path, output, "--max-new-tokens", "1"], "gpu_architecture")
assert result.returncode != 0 and "GPU architecture mismatch" in result.stderr, failure_detail(
    result, "gpu_architecture"
)
assert not output.exists()
report["rejections"].append("gpu_architecture")

weight_cases = (
    (
        "missing_shared_weights_contract",
        "Shared weights metadata is required",
        "Unsupported package metadata: shared_weights",
    ),
    (
        "mixed_shared_weights_contract",
        "Shared weights contracts differ",
        "Package metadata differs: shared_weights",
    ),
    (
        "mixed_weight_archive_checksum",
        "Shared weights contracts differ",
        "Package metadata differs: shared_weights",
    ),
    (
        "wrong_weight_archive_checksum",
        "Shared weights archive checksum mismatch",
        "Shared weights archive checksum mismatch",
    ),
    (
        "missing_weight_archive",
        "Shared weights archive is missing",
        "Shared weights archive is missing",
    ),
    (
        "corrupt_weight_archive",
        "Shared weights archive checksum mismatch",
        "Shared weights archive checksum mismatch",
    ),
    (
        "missing_embedded_weight_metadata",
        "Shared weights archive metadata is invalid",
        "Shared weights archive metadata is invalid",
    ),
)
for name, python_message, native_message in weight_cases:
    damaged = out / name
    damaged.mkdir()
    metadata = {
        kind: json.loads((packages / (kind + ".json")).read_text())
        for kind in ("prefill", "decode")
    }
    if name == "missing_shared_weights_contract":
        for item in metadata.values():
            del item["shared_weights"]
    elif name == "mixed_shared_weights_contract":
        metadata["prefill"]["shared_weights"]["identity"] = "0" * 64
    elif name == "mixed_weight_archive_checksum":
        metadata["prefill"]["shared_weights"]["sha256"] = "0" * 64
    elif name == "wrong_weight_archive_checksum":
        for item in metadata.values():
            item["shared_weights"]["sha256"] = "0" * 64
    if name == "corrupt_weight_archive":
        (damaged / "weights.pt").write_bytes(b"corrupted weight archive")
    elif name == "missing_embedded_weight_metadata":
        archive = damaged / "weights.pt"
        torch.jit.save(torch.jit.script(torch.nn.Identity()), str(archive))
        for item in metadata.values():
            item["shared_weights"].update(bytes=archive.stat().st_size, sha256=sha256(archive))
    elif name != "missing_weight_archive":
        (damaged / "weights.pt").symlink_to((packages / "weights.pt").resolve())
    for kind, item in metadata.items():
        (damaged / (kind + ".json")).write_text(json.dumps(item))
        (damaged / (kind + ".pt2")).symlink_to((packages / (kind + ".pt2")).resolve())
    try:
        validate_shared_weights(damaged, metadata["prefill"], metadata["decode"])
    except (ValueError, FileNotFoundError) as error:
        assert python_message in str(error), (name, str(error))
    else:
        raise AssertionError(f"Python accepted invalid weight bundle: {name}")
    result = run_native(
        [damaged, capital_prompt, out / "mixed.ids"],
        name,
        environment=env | {"CUDA_VISIBLE_DEVICES": ""},
        timeout=20,
    )
    assert result.returncode != 0 and native_message in result.stderr, failure_detail(result, name)
    report["rejections"].append(name)

seed = ids_from(capital_prompt)
for length, budget in ((1, 3), (2, 3), (512, 3), (len(seed), 1)):
    prompt = (seed * (length // len(seed) + 1))[:length]
    stem = f"length-{length}-budget-{budget}"
    input_path, ids_path, trace_path = (out / f"{stem}.{ext}" for ext in ("input", "ids", "f32"))
    input_path.write_text(" ".join(map(str, prompt)))
    result = run_native(
        [packages, input_path, ids_path, "--max-new-tokens", budget, "--trace", trace_path], stem
    )
    assert result.returncode == 0, failure_detail(result, stem)
    generated = ids_from(ids_path)
    entry = dict(prompt_length=length, budget=budget, tokens=len(generated))
    report["boundaries"].append(entry)
    native_traces.append((entry, prompt, generated, trace_path, budget))
    print(report["boundaries"][-1], flush=True)

# Finish native workers before retaining the Python process's GPU weight pool.
weights = SharedWeights(packages, spec["shared_weights"])
prefill, decode = weights.load("prefill"), weights.load("decode")
for entry, prompt, generated, trace_path, budget in native_traces:
    compare_native_trace(prompt, generated, trace_path, budget)
    entry["logits_bitwise_equal"] = True

report.update(
    passed=True,
    generation_limit_pass=True,
    no_python_linkage=True,
    native_loader_no_python_library=True,
    sanitized_environment_pass=True,
)
(out / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
print(
    "PASS: native generation/AOTI parity, package/input rejection, limits and Python-free linkage",
    flush=True,
)
