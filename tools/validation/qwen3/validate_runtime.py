"""Validate the native Flash CLI, deployment contract and exact Python AOTI parity."""

import argparse
from functools import partial
import json
from pathlib import Path
import subprocess

import numpy as np
from safetensors.torch import save_file
import torch

from compiler.common.weights import SharedWeights
from compiler.common.artifacts import (
    MODEL_MANIFEST_NAME,
    default_build_dir,
    read_json,
    require,
    sha256,
)
from compiler.common.metadata import (
    METADATA_FILE,
    read_package_metadata,
    validate_metadata,
)
from tools.packages import validate_qwen3_metadata as validate_model_metadata
from tools.packages import validate_shared_weights
from compiler.qwen3.config import DEFAULT_MODEL
from tools.common import native_environment, prompt_chunks
from tools.prepare.common import encode_prompt, load_tokenizer, write_ids
from tools.validation.common.prompts import chat_prompts
from tools.validation.common.validation import read_ids as ids_from
from tools.validation.common.runner import ExplicitRuntime


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native", type=Path, default=default_build_dir() / "xinfer_aoti")
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--packages", type=Path, default=Path("artifacts/aoti"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/native-smoke"))
    return parser


def preflight(binary, packages, out):
    linked = subprocess.check_output(["ldd", str(binary)], text=True)
    (out / "ldd.txt").write_text(linked)
    assert "not found" not in linked
    assert "libpython" not in linked and "libtorch_python" not in linked
    metadata = read_package_metadata(packages, model_type="qwen3")
    validate_model_metadata(metadata)
    validate_shared_weights(packages, metadata)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    return metadata


def run_native(binary, out, report, arguments, name, *, environment, timeout=120):
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
def compare_native_trace(prompt, generated, trace_path, budget, *, metadata, prefill, decode):
    assert 1 <= len(generated) <= budget
    assert not any(token in metadata["eos_token_ids"] for token in generated[:-1])
    assert len(generated) == budget or generated[-1] in metadata["eos_token_ids"]
    rows = np.fromfile(trace_path, dtype="<f4").reshape(len(generated), metadata["vocab_size"])
    assert np.isfinite(rows).all()
    runtime = ExplicitRuntime(
        dict(prefill=prefill, decode=decode),
        metadata["state"],
        metadata["vocab_size"],
        metadata["cache_length"],
        logits_only=True,
        max_prefill=metadata["max_prefill_tokens"],
    )
    runtime.reset()
    tokens = prompt
    for token, row in zip(generated, rows):
        for chunk in prompt_chunks(tokens, runtime.max_prefill):
            scores = runtime.step(chunk)
        assert torch.isfinite(scores).all().item()
        assert scores.argmax(-1).item() == token
        assert scores.float().cpu().numpy()[0].tobytes() == row.tobytes()
        tokens = [token]


def prepare_tokenizer(model_dir, metadata):
    require(
        read_json(model_dir / MODEL_MANIFEST_NAME) == metadata["checkpoint"],
        "Checkpoint manifest differs",
    )
    tokenizer = load_tokenizer(model_dir)
    return tokenizer


def check_generation(packages, out, env, tokenizer, run, report, native_traces):
    prompt_directory = out / "prompts"
    prompt_directory.mkdir()
    prompt_paths = {}
    for case in sorted(chat_prompts(), key=lambda item: item["name"]):
        name = case["name"]
        prompt = encode_prompt(tokenizer, case["text"])
        prompt_path = prompt_directory / f"{name}.ids"
        write_ids(prompt_path, prompt)
        prompt_paths[name] = prompt_path
        run_env = env | ({"LD_DEBUG": "libs"} if name == "capital" else {})
        ids, trace = out / f"{name}.ids", out / f"{name}.f32"
        budget = 64
        result = run(
            [packages, prompt_path, ids, "--max-output-tokens", budget, "--trace", trace],
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
    return prompt_paths["capital"]


def reject_inputs(packages, out, spec, capital_prompt, env, run, report):
    bad = out / "invalid.ids"
    cases = [
        ("empty", "", [], f"Expected 1..{spec['cache_length']} input tokens"),
        ("negative", "-1", [], "Token ID out of range"),
        ("vocab", str(spec["vocab_size"]), [], "Token ID out of range"),
        ("fraction", "1.5", [], "Invalid integer"),
        ("overflow", "999999999999999999999999999999", [], "Invalid integer"),
        (
            "long_prompt",
            "1 " * (spec["cache_length"] + 1),
            [],
            f"Expected 1..{spec['cache_length']} input tokens",
        ),
        ("retired_prompt_limit", "1", ["--max-prompt-length", "128"], "Unknown option"),
        ("retired_output_limit", "1", ["--max-new-tokens", "1"], "Unknown option"),
        ("zero_budget", "1", ["--max-output-tokens", "0"], "Generation budget"),
        ("negative_budget", "1", ["--max-output-tokens", "-1"], "Generation budget"),
        (
            "context_budget",
            "1 " * spec["cache_length"],
            ["--max-output-tokens", "1"],
            "Generation budget",
        ),
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
        result = run(
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
    result = run(
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


def reject_metadata(packages, out, capital_prompt, env, run, report):
    # All invalid package contracts must fail before CUDA or weight loading.
    mixed = out / "invalid-packages"
    mixed.mkdir()
    package_cases = (
        ("legacy_eager_numerics", "Unsupported package metadata: eager_numerics"),
        ("odd_state_count", "Unsupported package metadata: state"),
        ("mismatched_kv_shapes", "K/V shapes differ"),
        ("invalid_max_prefill_tokens", "Unsupported package metadata: max_prefill_tokens"),
    )
    package_cases += tuple(
        ("legacy_field_" + field, "Unsupported package metadata: " + field)
        for field in (
            "entry_point",
            "token_range",
            "export_options",
            "tf32",
            "triton_enable_fp_fusion",
            "inductor_configs",
            "package_bytes",
            "package_sha256",
            "model_specific",
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
        metadata = json.loads((packages / METADATA_FILE).read_text())
        if name == "legacy_eager_numerics":
            metadata["eager_numerics"] = True
        elif name.startswith("legacy_field_"):
            metadata[name.removeprefix("legacy_field_")] = False
        elif name == "odd_state_count":
            metadata["state"].pop()
        elif name == "mismatched_kv_shapes":
            metadata["state"][0]["shape"][1] = 4
        elif name == "invalid_max_prefill_tokens":
            metadata["max_prefill_tokens"] = 513
        (mixed / METADATA_FILE).write_text(json.dumps(metadata))
        if name.startswith("legacy_") or name == "invalid_max_prefill_tokens":
            try:
                validate_metadata(metadata, model_type="qwen3")
            except ValueError as error:
                assert message in str(error), (name, str(error))
            else:
                raise AssertionError(f"Python accepted invalid package: {name}")
        result = run(
            [mixed, capital_prompt, out / "mixed.ids"],
            name,
            environment=env | {"CUDA_VISIBLE_DEVICES": ""},
            timeout=20,
        )
        assert result.returncode != 0 and message in result.stderr, failure_detail(result, name)
        report["rejections"].append(name)


def package_fixture(packages, out, name, metadata, with_weights=False):
    directory = out / name
    directory.mkdir()
    (directory / METADATA_FILE).write_text(json.dumps(metadata))
    for kind in ("prefill", "decode"):
        (directory / f"{kind}.pt2").symlink_to((packages / f"{kind}.pt2").resolve())
    if with_weights:
        (directory / "weights.safetensors").symlink_to((packages / "weights.safetensors").resolve())
    return directory


def metadata_with_sizes(packages):
    metadata = json.loads((packages / METADATA_FILE).read_text())
    for kind in ("prefill", "decode"):
        metadata["entries"][kind]["bytes"] = (packages / f"{kind}.pt2").stat().st_size
    return metadata


def reject_dispatch(packages, out, capital_prompt, env, run, report):
    # Dispatch failures must not require CUDA or an archive.
    for name, tag, message in (
        ("missing_model_type", None, "Unsupported model type"),
        ("unknown_model_type", "unknown", "Unsupported model type"),
    ):
        directory = out / name
        directory.mkdir()
        metadata = json.loads((packages / METADATA_FILE).read_text())
        metadata.pop("model_type", None)
        if tag is not None:
            metadata["model_type"] = tag
        (directory / METADATA_FILE).write_text(json.dumps(metadata))
        output = directory / "generated.ids"
        result = run(
            [directory, capital_prompt, output],
            name,
            environment=env | {"CUDA_VISIBLE_DEVICES": ""},
            timeout=20,
        )
        assert result.returncode != 0 and message in result.stderr, failure_detail(result, name)
        assert not output.exists()
        report["rejections"].append(name)


def reject_package_files(packages, out, capital_prompt, env, run, report):
    # Declared sizes must fail before archive validation, even with CUDA hidden.
    for name, kind, missing in (
        ("wrong_prefill_size", "prefill", False),
        ("wrong_decode_size", "decode", False),
        ("missing_prefill_size", "prefill", True),
        ("missing_decode_size", "decode", True),
    ):
        metadata = metadata_with_sizes(packages)
        if missing:
            del metadata["entries"][kind]["bytes"]
        else:
            metadata["entries"][kind]["bytes"] += 1
        directory = package_fixture(packages, out, name, metadata)
        output = directory / "generated.ids"
        message = (
            "Unsupported package metadata: entries" if missing else "Package file size mismatch"
        )
        result = run(
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
        metadata = metadata_with_sizes(packages)
        if missing:
            del metadata["entries"][kind]["sha256"]
        else:
            metadata["entries"][kind]["sha256"] = "0" * 64
        directory = package_fixture(packages, out, name, metadata)
        output = directory / "generated.ids"
        message = "Unsupported package metadata: entries" if missing else "checksum mismatch"
        result = run(
            [directory, capital_prompt, output],
            name,
            environment=env | {"CUDA_VISIBLE_DEVICES": ""},
            timeout=20,
        )
        assert result.returncode != 0 and message in result.stderr, failure_detail(result, name)
        assert not output.exists(), name
        report["rejections"].append(name)


def check_package_loading(packages, out, capital_prompt, run, report, native_traces):
    # Valid size metadata must still pass complete loading and native/Python parity.
    directory = package_fixture(
        packages, out, "declared_package_sizes", metadata_with_sizes(packages), with_weights=True
    )
    prompt_path = capital_prompt
    ids_path, trace_path = directory / "generated.ids", directory / "logits.f32"
    result = run(
        [directory, prompt_path, ids_path, "--max-output-tokens", "1", "--trace", trace_path],
        "declared_package_sizes",
    )
    assert result.returncode == 0, failure_detail(result, "declared_package_sizes")
    generated = ids_from(ids_path)
    entry = dict(contract="declared_package_sizes", tokens=len(generated))
    report.setdefault("package_controls", []).append(entry)
    native_traces.append((entry, ids_from(prompt_path), generated, trace_path, 1))

    # The complete, checksum-valid bundle must reject an incompatible actual GPU.
    metadata = metadata_with_sizes(packages)
    metadata["cuda_compute_capability"] = [99, 0]
    directory = package_fixture(packages, out, "gpu_architecture", metadata, with_weights=True)
    output = directory / "generated.ids"
    result = run([directory, prompt_path, output, "--max-output-tokens", "1"], "gpu_architecture")
    assert result.returncode != 0 and "GPU architecture mismatch" in result.stderr, failure_detail(
        result, "gpu_architecture"
    )
    assert not output.exists()
    report["rejections"].append("gpu_architecture")


def reject_weights(packages, out, capital_prompt, env, run, report):
    weight_cases = (
        (
            "missing_shared_weights_contract",
            "Shared weights metadata is required",
            "Unsupported package metadata: shared_weights",
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
        metadata = json.loads((packages / METADATA_FILE).read_text())
        if name == "missing_shared_weights_contract":
            del metadata["shared_weights"]
        elif name == "wrong_weight_archive_checksum":
            metadata["shared_weights"]["sha256"] = "0" * 64
        if name == "corrupt_weight_archive":
            (damaged / "weights.safetensors").write_bytes(b"corrupted weight archive")
        elif name == "missing_embedded_weight_metadata":
            archive = damaged / "weights.safetensors"
            save_file({}, str(archive))
            metadata["shared_weights"].update(bytes=archive.stat().st_size, sha256=sha256(archive))
        elif name != "missing_weight_archive":
            (damaged / "weights.safetensors").symlink_to(
                (packages / "weights.safetensors").resolve()
            )
        (damaged / METADATA_FILE).write_text(json.dumps(metadata))
        for kind in ("prefill", "decode"):
            (damaged / (kind + ".pt2")).symlink_to((packages / (kind + ".pt2")).resolve())
        try:
            validate_shared_weights(damaged, metadata)
        except (ValueError, FileNotFoundError) as error:
            assert python_message in str(error), (name, str(error))
        else:
            raise AssertionError(f"Python accepted invalid weight bundle: {name}")
        result = run(
            [damaged, capital_prompt, out / "mixed.ids"],
            name,
            environment=env | {"CUDA_VISIBLE_DEVICES": ""},
            timeout=20,
        )
        assert result.returncode != 0 and native_message in result.stderr, failure_detail(
            result, name
        )
        report["rejections"].append(name)


def check_boundaries(packages, out, spec, capital_prompt, run, report, native_traces):
    seed = ids_from(capital_prompt)
    boundary_requests = [(length, 3) for length in (1, 2, 511, 512, 513, 8191, 8192, 8193)]
    boundary_requests += [(len(seed), 1), (128, 1)]
    for length, budget in boundary_requests:
        if length + budget > spec["cache_length"]:
            continue
        prompt = (seed * (length // len(seed) + 1))[:length]
        stem = f"length-{length}-budget-{budget}"
        input_path, ids_path, trace_path = (
            out / f"{stem}.{ext}" for ext in ("input", "ids", "f32")
        )
        input_path.write_text(" ".join(map(str, prompt)))
        result = run(
            [packages, input_path, ids_path, "--max-output-tokens", budget, "--trace", trace_path],
            stem,
        )
        assert result.returncode == 0, failure_detail(result, stem)
        generated = ids_from(ids_path)
        entry = dict(prompt_length=length, budget=budget, tokens=len(generated))
        report["boundaries"].append(entry)
        native_traces.append((entry, prompt, generated, trace_path, budget))
        print(report["boundaries"][-1], flush=True)


def compare_native_traces(native_traces, metadata, prefill, decode):
    for entry, prompt, generated, trace_path, budget in native_traces:
        compare_native_trace(
            prompt, generated, trace_path, budget, metadata=metadata, prefill=prefill, decode=decode
        )
        entry["logits_bitwise_equal"] = True


def write_report(out, report):
    report.update(
        passed=True,
        generation_limit_pass=True,
        no_python_linkage=True,
        native_loader_no_python_library=True,
        sanitized_environment_pass=True,
    )
    (out / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(
        "PASS: native generation/AOTI parity, package/input rejection, limits "
        "and Python-free linkage",
        flush=True,
    )


def validate_runtime(*, native, packages, model_dir, output):
    """Check the shipping CLI before loading Python AOTI for exact comparisons."""
    binary, out = native.resolve(), output
    out.mkdir(parents=True)
    env = native_environment()
    report = dict(fixtures=[], rejections=[], boundaries=[], native_processes=[])
    metadata = preflight(binary, packages, out)
    native_traces = []
    run = partial(run_native, binary, out, report, environment=env)
    tokenizer = prepare_tokenizer(model_dir, metadata)
    capital_prompt = check_generation(packages, out, env, tokenizer, run, report, native_traces)
    reject_inputs(packages, out, metadata, capital_prompt, env, run, report)
    reject_metadata(packages, out, capital_prompt, env, run, report)
    reject_dispatch(packages, out, capital_prompt, env, run, report)
    reject_package_files(packages, out, capital_prompt, env, run, report)
    check_package_loading(packages, out, capital_prompt, run, report, native_traces)
    reject_weights(packages, out, capital_prompt, env, run, report)
    check_boundaries(packages, out, metadata, capital_prompt, run, report, native_traces)

    # Finish native workers before retaining the Python process's GPU weight pool.
    weights = SharedWeights(packages, metadata["shared_weights"])
    prefill, decode = weights.load("prefill"), weights.load("decode")
    compare_native_traces(native_traces, metadata, prefill, decode)
    write_report(out, report)
    return report


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error("Output directory already exists; choose a fresh --output")
    validate_runtime(
        native=args.native, packages=args.packages, model_dir=args.model_dir, output=args.output
    )


if __name__ == "__main__":
    main()
