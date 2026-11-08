"""Validate the native text runtime against Python AOTI; HF numerical acceptance is separate."""

import argparse
import copy
import gc
import json
import os
from pathlib import Path
import subprocess

import torch
from transformers import AutoTokenizer

from compiler.qwen35.config import ARTIFACT_DIR, TEXT_DIR
from compiler.common.artifacts import MODEL_MANIFEST_NAME, default_build_dir, sha256
from compiler.common.metadata import RUNTIME_ABI_VERSION, read_package_metadata
from compiler.qwen35.packages import validate_model_metadata
from compiler.common.weights import validate_shared_weights
from tools.common import prompt_chunks, run
from tools.prepare.common import encode_prompt
from compiler.qwen35.model import configure_text
from tools.validation.qwen35.common import fixtures
from tools.validation.common.prompts import chat_prompts
from tools.validation.common.runner import greedy, load_aoti_runtime, run_stage
from tools.validation.common.native import run_native
from tools.validation.common.chain import Policy, compare_traces, new_report
from tools.validation.common.context import capture_long_state, long_context_cases
from tools.validation.common.validation import compare_file


def reject_invalid(args, env, metadata):
    metadata = copy.deepcopy(metadata)
    directory = args.output / "rejections"
    directory.mkdir()
    capacity = metadata["decode"]["cache_length"]
    cases = [
        ("empty", "", "1", f"Expected 1..{capacity}"),
        ("oversized", "42 " * (capacity + 1), "1", f"Expected 1..{capacity}"),
        ("negative_id", "-1", "1", "Token ID out of range"),
        ("large_id", str(metadata["decode"]["vocab_size"]), "1", "Token ID out of range"),
        ("not_integer", "42junk", "1", "Invalid integer"),
        ("zero_budget", "42", "0", "Generation budget"),
        ("negative_budget", "42", "-1", "Generation budget"),
        ("overflow_budget", "42", str(capacity), "Generation budget"),
        ("malformed_budget", "42", "2junk", "Invalid integer"),
    ]
    passed = []
    for name, text, limit, error in cases:
        prompt, output = directory / f"{name}.ids", directory / f"{name}.out"
        prompt.write_text(text)
        run(
            [args.cli, args.packages, prompt, output, "--max-output-tokens", limit],
            env,
            directory / f"{name}.log",
            success=False,
            error=error,
        )
        assert not output.exists()
        passed.append(name)
    for name, flag in (
        ("retired_prompt_limit", "--max-prompt-length"),
        ("retired_output_limit", "--max-new-tokens"),
    ):
        prompt, output = directory / f"{name}.ids", directory / f"{name}.out"
        prompt.write_text("42\n")
        run(
            [args.cli, args.packages, prompt, output, flag, "1"],
            env,
            directory / f"{name}.log",
            success=False,
            error="Unknown option",
        )
        assert not output.exists()
        passed.append(name)
    prompt = directory / "valid.ids"
    prompt.write_text("42\n")
    run(
        [args.cli, args.packages, prompt, prompt],
        env,
        directory / "overwrite.log",
        success=False,
        error="Output already exists",
    )
    assert prompt.read_text() == "42\n"
    passed.append("overwrite")
    mutations = [
        (
            "missing_schema_version",
            lambda m: [v.pop("schema_version") for v in m.values()],
            "Unsupported package metadata schema version",
        ),
        (
            "unknown_schema_version",
            lambda m: [v.update(schema_version=999) for v in m.values()],
            "Unsupported package metadata schema version",
        ),
        (
            "mixed_schema_version",
            lambda m: m["prefill"].update(schema_version=999),
            "Unsupported package metadata schema version",
        ),
        (
            "mixed_model_types",
            lambda m: m["prefill"].update(model_type="qwen3"),
            "Package model types differ",
        ),
        (
            "missing_prefill_model_type",
            lambda m: m["prefill"].pop("model_type"),
            "Package model types differ",
        ),
        (
            "missing_decode_model_type",
            lambda m: m["decode"].pop("model_type"),
            "Package model types differ",
        ),
        (
            "unknown_model_type",
            lambda m: [v.update(model_type="unknown") for v in m.values()],
            "Unsupported model type",
        ),
        (
            "mixed_checkpoint",
            lambda m: m["prefill"]["checkpoint"].update(revision="different"),
            "Package metadata differs: checkpoint",
        ),
        (
            "mixed_state",
            lambda m: m["prefill"]["state"][0]["shape"].__setitem__(-1, 5),
            "Package metadata differs: state",
        ),
        (
            "legacy_prefill",
            lambda m: m["prefill"].update(eager_numerics=True),
            "Unsupported package metadata: eager_numerics",
        ),
        (
            "legacy_decode",
            lambda m: m["decode"].update(eager_numerics=True),
            "Unsupported package metadata: eager_numerics",
        ),
        (
            "legacy_pair",
            lambda m: [v.update(eager_numerics=True) for v in m.values()],
            "Unsupported package metadata: eager_numerics",
        ),
        (
            "mixed_packing",
            lambda m: m["prefill"]["model_specific"].update(
                linear_packing={"enabled": "different"}
            ),
            "Package metadata differs: model_specific",
        ),
        (
            "entry_point",
            lambda m: m["prefill"].update(entry_point="decode"),
            "Unsupported package metadata: entry_point",
        ),
        (
            "torch_version",
            lambda m: [v.update(torch_version="0.0.0") for v in m.values()],
            "Torch version mismatch",
        ),
        (
            "state_dtype",
            lambda m: [v["state"][0].update(dtype="float32") for v in m.values()],
            "Unsupported package metadata: state",
        ),
        (
            "state_order",
            lambda m: [v["state"].reverse() for v in m.values()],
            "Invalid state order or dtype",
        ),
        (
            "gpu_arch",
            lambda m: [v.update(cuda_compute_capability=[99, 0]) for v in m.values()],
            "GPU architecture mismatch",
        ),
    ]
    mutations += [
        (
            "missing_runtime_abi",
            lambda m: [v.pop("runtime_abi_version") for v in m.values()],
            "Unsupported runtime ABI version",
        ),
        (
            "unknown_runtime_abi",
            lambda m: [v.update(runtime_abi_version=RUNTIME_ABI_VERSION + 1) for v in m.values()],
            "Unsupported runtime ABI version",
        ),
        (
            "mixed_runtime_abi",
            lambda m: m["prefill"].update(runtime_abi_version=RUNTIME_ABI_VERSION + 1),
            "Unsupported runtime ABI version",
        ),
        (
            "boolean_runtime_abi",
            lambda m: [v.update(runtime_abi_version=True) for v in m.values()],
            "Unsupported runtime ABI version",
        ),
        (
            "float_runtime_abi",
            lambda m: [
                v.update(runtime_abi_version=float(RUNTIME_ABI_VERSION)) for v in m.values()
            ],
            "Unsupported runtime ABI version",
        ),
        (
            "mixed_export_options",
            lambda m: m["prefill"]["export_options"].update(backed_size_oblivious=True),
            "Package metadata differs: export_options",
        ),
        (
            "invalid_export_options",
            lambda m: [v["export_options"].update(backed_size_oblivious=True) for v in m.values()],
            "Unsupported package metadata: export_options",
        ),
        (
            "missing_export_options",
            lambda m: [v.pop("export_options") for v in m.values()],
            "Unsupported package metadata: export_options",
        ),
    ]
    for field in (
        "batch_size",
        "dtype",
        "input_mutation",
        "attention_backend",
        "custom_attention",
        "cache_input",
        "kv_extent",
    ):
        mutations.append(
            (
                "legacy_" + field,
                lambda m, field=field: [v.update({field: False}) for v in m.values()],
                "Unsupported package metadata: " + field,
            )
        )
    mutations += [
        (
            "mixed_delta_threshold",
            lambda m: m["prefill"]["model_specific"].update(delta_short_threshold=98),
            "Package metadata differs: model_specific",
        ),
        (
            "mixed_delta_algorithm",
            lambda m: m["prefill"]["model_specific"].update(
                delta_algorithm={"prefill": "different"}
            ),
            "Package metadata differs: model_specific",
        ),
        (
            "mixed_delta_tile",
            lambda m: m["prefill"]["model_specific"].update(delta_transform_tile=64),
            "Package metadata differs: model_specific",
        ),
        (
            "prefill_range",
            lambda m: m["prefill"].update(token_range=[2, 513]),
            "Unsupported package metadata: token_range",
        ),
        (
            "decode_range",
            lambda m: m["decode"].update(token_range=[1, 2]),
            "Unsupported package metadata: token_range",
        ),
        (
            "mixed_backend",
            lambda m: m["prefill"]["model_specific"].update(delta_backend="invalid"),
            "Package metadata differs: model_specific",
        ),
        (
            "unknown_backend",
            lambda m: [v["model_specific"].update(delta_backend="invalid") for v in m.values()],
            "Unsupported DeltaNet backend",
        ),
        (
            "mixed_shared_weights",
            lambda m: m["prefill"]["shared_weights"].update(identity="0" * 64),
            "Package metadata differs: shared_weights",
        ),
        (
            "missing_shared_weights",
            lambda m: [v.pop("shared_weights") for v in m.values()],
            "Unsupported package metadata: shared_weights",
        ),
        (
            "mixed_archive_checksum",
            lambda m: m["prefill"]["shared_weights"].update(sha256="0" * 64),
            "Package metadata differs: shared_weights",
        ),
        (
            "wrong_archive_checksum",
            lambda m: [v["shared_weights"].update(sha256="0" * 64) for v in m.values()],
            "Shared weights archive checksum mismatch",
        ),
    ]
    if "delta_short_threshold" in metadata["decode"]["model_specific"]:
        mutations.append(
            (
                "missing_delta_threshold",
                lambda m: m["prefill"]["model_specific"].pop("delta_short_threshold"),
                "Package metadata differs: model_specific",
            )
        )
    for name, mutate, error in mutations:
        packages = directory / name
        packages.mkdir()
        modified = copy.deepcopy(metadata)
        mutate(modified)
        for kind, meta in modified.items():
            (packages / f"{kind}.json").write_text(json.dumps(meta))
            (packages / f"{kind}.pt2").symlink_to((args.packages / f"{kind}.pt2").resolve())
        (packages / "weights.pt").symlink_to((args.packages / "weights.pt").resolve())
        output = packages / "generated.ids"
        run(
            [args.cli, packages, prompt, output],
            env,
            directory / f"{name}.log",
            success=False,
            error=error,
        )
        assert not output.exists()
        passed.append(name)
    for name, error in (
        ("missing_weights_archive", "Shared weights archive is missing"),
        ("corrupt_weights_archive", "Shared weights archive checksum mismatch"),
        ("missing_embedded_weights_metadata", "Shared weights archive metadata is invalid"),
    ):
        packages = directory / name
        packages.mkdir()
        modified = copy.deepcopy(metadata)
        if name == "corrupt_weights_archive":
            (packages / "weights.pt").write_bytes(b"corrupt shared weights\n")
        elif name == "missing_embedded_weights_metadata":
            archive = packages / "weights.pt"
            torch.jit.save(torch.jit.script(torch.nn.Identity()), str(archive))
            for item in modified.values():
                item["shared_weights"].update(bytes=archive.stat().st_size, sha256=sha256(archive))
        for kind, meta in modified.items():
            (packages / f"{kind}.json").write_text(json.dumps(meta))
            (packages / f"{kind}.pt2").symlink_to((args.packages / f"{kind}.pt2").resolve())
        output = packages / "generated.ids"
        run(
            [args.cli, packages, prompt, output],
            env,
            directory / f"{name}.log",
            success=False,
            error=error,
        )
        assert not output.exists()
        passed.append(name)
    return passed


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=TEXT_DIR)
    parser.add_argument("--packages", type=Path, default=ARTIFACT_DIR / "text")
    parser.add_argument(
        "--binary", type=Path, default=default_build_dir("qwen35") / "xinfer_validate"
    )
    parser.add_argument("--cli", type=Path, default=default_build_dir("qwen35") / "xinfer_aoti")
    parser.add_argument("--output", type=Path, default=ARTIFACT_DIR / "runtime-validation")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    metadata = read_package_metadata(args.packages, model_type="qwen35")
    for entry in metadata.values():
        validate_model_metadata(entry)
    validate_shared_weights(args.packages, metadata["prefill"], metadata["decode"])
    configure_text()
    for kind, meta in metadata.items():
        assert meta["package_sha256"] == sha256(args.packages / f"{kind}.pt2")
        assert meta["checkpoint"]["manifest"] == json.loads(
            (args.model_dir / MODEL_MANIFEST_NAME).read_text()
        )
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, local_files_only=True)
    assert metadata["prefill"]["model_specific"]["delta_backend"] == "triton_op"
    capacity = metadata["decode"]["cache_length"]
    cases = [case for case in fixtures(tokenizer) if sum(map(len, case["steps"])) <= capacity]
    seed = cases[1]["steps"][0]
    for length in (2, 63, 65, 96, 97, 98, 129, 511):
        prompt = (seed * (length // len(seed) + 1))[:length]
        if length + 1 <= capacity:
            cases.append(dict(name=f"dispatch_{length}", steps=[prompt, [42]]))
    capacity = metadata["decode"]["cache_length"]
    maximum = metadata["prefill"]["token_range"][1]
    stream = (seed * (capacity // len(seed) + 1))[:capacity]
    cases.append(
        dict(
            name="context_end",
            steps=(
                [stream[:512], stream[512:960], stream[960:1023], stream[1023:]]
                if capacity == 1024
                else prompt_chunks(stream[:-1], maximum) + [stream[-1:]]
            ),
            check_overflow=True,
            long_context=capacity > 1024,
        )
    )
    cases.extend(long_context_cases(seed, capacity, maximum))
    if capacity >= 8196:
        cases.append(dict(name="long_prompt_8193_generate", prompt=stream[:8193], limit=3))
    for prompt in chat_prompts("qwen35"):
        cases.append(
            dict(
                name=prompt["name"],
                prompt=encode_prompt(tokenizer, prompt["text"]),
                text=prompt["text"],
                limit=24,
            )
        )
    capital = next(c for c in cases if c["name"] == "capital")
    cases += [dict(capital, name="capital_repeat"), dict(capital, name="limit_one", limit=1)]
    eos = metadata["decode"]["eos_token_ids"]
    for case in list(cases):
        if case["name"] in ("long_prompt_513", "long_prompt_8191", "long_prompt_8192"):
            prompt = [token for chunk in case["steps"][:-2] for token in chunk]
            cases.append(dict(name=case["name"] + "_generate", prompt=prompt, limit=3))
        if case["name"] == "capital_repeat":
            case["same_as"] = "capital"
        if case["name"] == "repeat":
            case["same_as"] = "prefill_decode"
        if case["name"] == "mixed_dispatch_repeat":
            case["same_as"] = "mixed_dispatch"
    teacher = [case for case in cases if "steps" in case]
    chats = [case for case in cases if "prompt" in case]

    def capture(case, step):
        return capture_long_state(case, step) if case.get("long_context") else True

    keys = {
        (case["name"], step)
        for case in teacher
        for step in range(len(case["steps"]))
        if capture(case, step)
    }
    runtime = load_aoti_runtime(
        args.packages, model_type="qwen35", specs=metadata["decode"]["state"]
    )
    print("Python AOTI reference", flush=True)
    try:
        expected = run_stage(runtime, teacher, chats, metadata["decode"]["state"], capture, eos)
        capital_result = expected.greedy["capital"]
        forced = dict(capital, name="forced_eos", eos=[capital_result["ids"][0]])
        chats.append(forced)
        expected.greedy["forced_eos"] = greedy(
            runtime, forced["prompt"], forced["limit"], forced["eos"]
        )
    finally:
        runtime.close()
        del runtime
        gc.collect()
        torch.cuda.empty_cache()
    env = {
        k: v
        for k, v in os.environ.items()
        if k not in ("VIRTUAL_ENV", "PYTHONPATH", "PYTHONHOME", "LD_LIBRARY_PATH")
    }
    env.update(PATH="/usr/bin:/bin", PYTHONHOME="/nonexistent-python-home")
    print("Fresh native process", flush=True)
    actual, execution = run_native(
        args.binary, args.packages, args.output, teacher, chats, metadata, state_keys=keys
    )
    report = new_report("qwen35", metadata)
    compare_traces(
        report,
        "cpp_aoti",
        "python_aoti",
        actual,
        expected,
        metadata["decode"]["state"],
        Policy(),
        exact=True,
        state_keys=keys,
        greedy_state_names={case["name"] for case in chats},
    )
    # Forced-EOS must stop without feeding the generated token back.
    forced_actual = actual.greedy["forced_eos"]
    assert forced_actual["position"] == len(capital["prompt"])
    report.update(
        comparison="C++ vs Python AOTI",
        hf_numerical_acceptance="evaluated separately by tools.validation.validate",
        cases=execution["cases"],
        native_execution=execution,
        teacher_tensors=len(expected.logits)
        + sum(len(state) for state in expected.states.values()),
        greedy_logit_rows=sum(len(value["rows"]) for value in expected.greedy.values()),
        greedy_state_tensors=sum(len(value["state"]) for value in expected.greedy.values()),
        binary_sha256={p.name: sha256(p) for p in (args.binary, args.cli)},
        state_snapshot_policy="complete functional state and populated KV prefix; future fill checked separately",
    )
    assert not report["failures"], report["failures"]
    prompt_path, ids_path, trace_path = [
        args.output / name for name in ("prompt.ids", "generated.ids", "cli.f32")
    ]
    prompt_path.write_text(" ".join(map(str, capital["prompt"])) + "\n")
    run(
        [
            args.cli.resolve(),
            args.packages.resolve(),
            prompt_path.resolve(),
            ids_path.resolve(),
            "--max-output-tokens",
            capital["limit"],
            "--trace",
            trace_path.resolve(),
        ],
        env,
        args.output / "cli.log",
    )
    assert [int(value) for value in ids_path.read_text().split()] == capital_result["ids"]
    compare_file(trace_path, capital_result["rows"])
    for binary in (args.binary, args.cli):
        linked = subprocess.check_output(["ldd", str(binary.resolve())], env=env, text=True)
        (args.output / f"{binary.name}.ldd").write_text(linked)
        assert "libpython" not in linked and "libtorch_python" not in linked
    print("Input/package rejection checks", flush=True)
    report["rejections"] = reject_invalid(args, env, metadata)
    report.update(
        passed=True,
        native_bitwise=True,
        native_runtime_passed=True,
        no_python_library=True,
        repeated_requests_bitwise=True,
        cli_bitwise=True,
        dispatch_counts_checked=True,
    )
    (args.output / "summary.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n"
    )
    print(
        json.dumps(
            {
                k: report[k]
                for k in (
                    "native_runtime_passed",
                    "teacher_tensors",
                    "greedy_logit_rows",
                    "greedy_state_tensors",
                    "hf_numerical_acceptance",
                )
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
