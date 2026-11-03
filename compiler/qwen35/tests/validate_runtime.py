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
from compiler.common.artifacts import NATIVE_RUNTIME_SOURCES, sha256, ROOT
from compiler.common.metadata import read_package_metadata
from compiler.qwen35.packages import validate_model_metadata
from compiler.common.attention import load_attention_ops, native_attention_library
from compiler.qwen35.prepare import encode_prompt
from compiler.qwen35.model import configure_text
from compiler.qwen35.tests.common import fixtures
from compiler.qwen35.reference import PythonRuntime
from compiler.common.tests.validation import cpu_copy, compare_file
from compiler.common.artifacts import run


def reject_invalid(args, env, metadata):
    metadata = copy.deepcopy(metadata)
    directory = args.output / "rejections"
    directory.mkdir()
    cases = [("empty", "", "1", "Expected 1..512"),
             ("oversized", "42 " * 513, "1", "Expected 1..512"),
             ("negative_id", "-1", "1", "Token ID out of range"),
             ("large_id", str(metadata["decode"]["vocab_size"]), "1", "Token ID out of range"),
             ("not_integer", "42junk", "1", "Invalid integer"),
             ("zero_budget", "42", "0", "Generation budget"),
             ("negative_budget", "42", "-1", "Generation budget"),
             ("overflow_budget", "42", "1024", "Generation budget"),
             ("malformed_budget", "42", "2junk", "Invalid integer")]
    passed = []
    for name, text, limit, error in cases:
        prompt, output = directory / f"{name}.ids", directory / f"{name}.out"
        prompt.write_text(text)
        run([args.cli, args.packages, prompt, output, "--max-new-tokens", limit],
            env, directory / f"{name}.log", success=False, error=error)
        assert not output.exists()
        passed.append(name)
    prompt = directory / "valid.ids"
    prompt.write_text("42\n")
    run([args.cli, args.packages, prompt, prompt], env, directory / "overwrite.log",
        success=False, error="Output already exists")
    assert prompt.read_text() == "42\n"
    passed.append("overwrite")
    mutations = [
        ("missing_schema_version", lambda m: [v.pop("schema_version") for v in m.values()],
         "Unsupported package metadata schema version"),
        ("unknown_schema_version", lambda m: [v.update(schema_version=999) for v in m.values()],
         "Unsupported package metadata schema version"),
        ("mixed_schema_version", lambda m: m["prefill"].update(schema_version=999),
         "Unsupported package metadata schema version"),
        ("mixed_model_types", lambda m: m["prefill"].update(model_type="qwen3"), "Package model types differ"),
        ("missing_prefill_model_type", lambda m: m["prefill"].pop("model_type"), "Package model types differ"),
        ("missing_decode_model_type", lambda m: m["decode"].pop("model_type"), "Package model types differ"),
        ("unknown_model_type", lambda m: [v.update(model_type="unknown") for v in m.values()], "Unsupported model type"),
        ("mixed_checkpoint", lambda m: m["prefill"]["checkpoint"].update(revision="different"), "Package metadata differs: checkpoint"),
        ("mixed_state", lambda m: m["prefill"]["state"][0]["shape"].__setitem__(-1, 5), "Package metadata differs: state"),
        ("legacy_prefill", lambda m: m["prefill"].update(eager_numerics=True), "Unsupported package metadata: eager_numerics"),
        ("legacy_decode", lambda m: m["decode"].update(eager_numerics=True), "Unsupported package metadata: eager_numerics"),
        ("legacy_pair", lambda m: [v.update(eager_numerics=True) for v in m.values()], "Unsupported package metadata: eager_numerics"),
        ("mixed_packing", lambda m: m["prefill"]["model_specific"].update(linear_packing={"enabled": "different"}),
         "Package metadata differs: model_specific"),
        ("entry_point", lambda m: m["prefill"].update(entry_point="decode"), "Unsupported package metadata: entry_point"),
        ("torch_version", lambda m: [v.update(torch_version="0.0.0") for v in m.values()], "Torch version mismatch"),
        ("state_dtype", lambda m: [v["state"][0].update(dtype="float32") for v in m.values()], "Unsupported package metadata: state"),
        ("abi_order", lambda m: [v["input_order"].reverse() for v in m.values()], "Unsupported package metadata: input_order"),
        ("gpu_arch", lambda m: [v.update(cuda_compute_capability=[99, 0]) for v in m.values()], "GPU architecture mismatch"),
    ]
    for name, field, invalid in (
        ("attention_backend", "attention_backend", "efficient_sdpa"),
        ("cache_contract", "cache_input", "full_capacity"),
        ("input_mutation", "input_mutation", False),
    ):
        mutations.append(("mixed_" + name, lambda m, field=field, invalid=invalid: m["prefill"].update({field: invalid}),
                          "Package metadata differs: " + field))
        mutations.append(("invalid_" + name, lambda m, field=field, invalid=invalid: [v.update({field: invalid}) for v in m.values()],
                          "Unsupported package metadata: " + field))
    for name, field, update in (
        ("attention_contract", "custom_attention", {"abi_version": 0}),
        ("cache_extent", "kv_extent", {"data_read": True}),
        ("export_options", "export_options", {"backed_size_oblivious": True}),
    ):
        mutations.append(("mixed_" + name, lambda m, field=field, update=update: m["prefill"][field].update(update),
                          "Package metadata differs: " + field))
        mutations.append(("invalid_" + name, lambda m, field=field, update=update: [v[field].update(update) for v in m.values()],
                          "Unsupported package metadata: " + field))
    for field in ("custom_attention", "cache_input", "kv_extent"):
        mutations.append(("missing_" + field, lambda m, field=field: [v.pop(field) for v in m.values()],
                          "Unsupported package metadata: " + field))
    mutations += [
        ("mixed_delta_threshold", lambda m: m["prefill"]["model_specific"].update(delta_short_threshold=98),
         "Package metadata differs: model_specific"),
        ("mixed_delta_algorithm", lambda m: m["prefill"]["model_specific"].update(delta_algorithm={"prefill": "different"}),
         "Package metadata differs: model_specific"),
        ("mixed_delta_tile", lambda m: m["prefill"]["model_specific"].update(delta_transform_tile=64),
         "Package metadata differs: model_specific"),
        ("prefill_range", lambda m: m["prefill"].update(token_range=[2, 513]), "Unsupported package metadata: token_range"),
        ("decode_range", lambda m: m["decode"].update(token_range=[1, 2]), "Unsupported package metadata: token_range"),
        ("mixed_backend", lambda m: m["prefill"]["model_specific"].update(delta_backend="invalid"),
         "Package metadata differs: model_specific"),
        ("unknown_backend", lambda m: [v["model_specific"].update(delta_backend="invalid") for v in m.values()],
         "Unsupported DeltaNet backend"),
        ("mixed_shared_weights", lambda m: m["prefill"]["shared_weights"].update(identity="0" * 64),
         "Package metadata differs: shared_weights"),
        ("missing_shared_weights", lambda m: [v.pop("shared_weights") for v in m.values()],
         "Unsupported package metadata: shared_weights"),
        ("mixed_archive_checksum", lambda m: m["prefill"]["shared_weights"].update(sha256="0" * 64),
         "Package metadata differs: shared_weights"),
        ("wrong_archive_checksum", lambda m: [v["shared_weights"].update(sha256="0" * 64) for v in m.values()],
         "Shared weights archive checksum mismatch"),
    ]
    if "delta_short_threshold" in metadata["decode"]["model_specific"]:
        mutations.append(("missing_delta_threshold", lambda m: m["prefill"]["model_specific"].pop("delta_short_threshold"),
                          "Package metadata differs: model_specific"))
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
        run([args.cli, packages, prompt, output], env, directory / f"{name}.log", success=False, error=error)
        assert not output.exists()
        passed.append(name)
    for name, error in (("missing_weights_archive", "Shared weights archive is missing"),
                        ("corrupt_weights_archive", "Shared weights archive checksum mismatch"),
                        ("missing_embedded_weights_metadata", "Shared weights archive metadata is invalid")):
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
        run([args.cli, packages, prompt, output], env, directory / f"{name}.log",
            success=False, error=error)
        assert not output.exists()
        passed.append(name)
    return passed


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=TEXT_DIR)
    parser.add_argument("--packages", type=Path, default=ARTIFACT_DIR / "text")
    parser.add_argument("--binary", type=Path, default=Path("work/qwen35/build/xinfer_validate"))
    parser.add_argument("--cli", type=Path, default=Path("work/qwen35/build/xinfer_aoti"))
    parser.add_argument("--output", type=Path, default=ARTIFACT_DIR / "runtime-validation")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    metadata = read_package_metadata(args.packages, model_type="qwen35")
    for entry in metadata.values():
        validate_model_metadata(entry)
    attention_library = native_attention_library(args.binary)
    if attention_library is None:
        raise ValueError("Qwen3.5 runtime is missing its required native attention library; rebuild the runtime")
    os.environ["XINFER_ATTENTION_OPS_LIBRARY"] = str(attention_library)
    load_attention_ops(attention_library)
    configure_text()
    for kind, meta in metadata.items():
        assert meta["package_sha256"] == sha256(args.packages / f"{kind}.pt2")
        assert meta["checkpoint"]["manifest"] == json.loads((args.model_dir / "manifest.json").read_text())
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, local_files_only=True)
    assert metadata["prefill"]["model_specific"]["delta_backend"] == "triton_op"
    cases = fixtures(tokenizer)
    seed = cases[1]["steps"][0]
    for length in (2, 63, 65, 96, 97, 98, 129, 511):
        prompt = (seed * (length // len(seed) + 1))[:length]
        cases.append(dict(name=f"dispatch_{length}", steps=[prompt, [42]]))
    stream = seed * 16
    cases.append(dict(name="context_end", steps=[stream[:512], stream[512:960], stream[960:1023], stream[1023:]],
                      check_overflow=True))
    for name, text in (("capital", "What is the capital of France? Answer briefly."),
                       ("arithmetic", "What is 17 + 25? Answer with only the number."),
                       ("chinese", "用一句话解释什么是编译器。")):
        cases.append(dict(name=name, prompt=encode_prompt(tokenizer, text), text=text, limit=24))
    capital = next(c for c in cases if c["name"] == "capital")
    cases += [dict(capital, name="capital_repeat"), dict(capital, name="limit_one", limit=1)]
    eos = metadata["decode"]["eos_token_ids"]
    runtime = PythonRuntime(args.packages, metadata["decode"]["state"])
    expected = {}
    print("Python AOTI reference", flush=True)
    for case in cases:
        runtime.reset()
        rows, ids = [], []
        if "prompt" in case:
            logits = runtime.step(case["prompt"])
            for i in range(case["limit"]):
                rows.append(logits.cpu().float())
                token = logits.argmax(-1).item()
                ids.append(token)
                if token in case.get("eos", eos) or i + 1 == case["limit"]:
                    break
                logits = runtime.step([token])
            assert runtime.position == len(case["prompt"]) + len(ids) - 1
            expected[case["name"]] = dict(ids=ids, rows=rows, state=cpu_copy(runtime.state), position=runtime.position)
            if case["name"] == "capital":
                cases.append(dict(capital, name="forced_eos", eos=[ids[0]]))
        else:
            for tokens in case["steps"]:
                rows.append(cpu_copy((runtime.step(tokens), *runtime.state)))
            assert runtime.position == sum(map(len, case["steps"]))
            expected[case["name"]] = dict(rows=rows, position=runtime.position)
        print(case["name"], "position", runtime.position, flush=True)
    del runtime, logits
    gc.collect()
    torch.cuda.empty_cache()
    manifest = args.output / "cases.json"
    manifest.write_text(json.dumps(dict(cases=cases), indent=2, ensure_ascii=False) + "\n")
    env = {k: v for k, v in os.environ.items() if k not in ("VIRTUAL_ENV", "PYTHONPATH", "PYTHONHOME", "LD_LIBRARY_PATH")}
    env.update(PATH="/usr/bin:/bin", PYTHONHOME="/nonexistent-python-home")
    print("Fresh native process", flush=True)
    run([args.binary.resolve(), args.packages.resolve(), manifest.resolve(), (args.output / "native").resolve()],
        env, args.output / "native.log")
    native = json.loads((args.output / "native/native.json").read_text())
    assert native["loader_pairs"] == 1 and native["no_python_library"] and native["attention_library"]
    assert len(native["cases"]) == len(cases)
    report = dict(comparison="C++ vs Python AOTI", hf_numerical_acceptance="open: Q35-NUM-001 and step-3 state failures",
                  delta_backend=metadata["prefill"]["model_specific"]["delta_backend"],
                  attention_backend=metadata["prefill"]["attention_backend"],
                  linear_packing=metadata["prefill"]["model_specific"].get("linear_packing"),
                  shared_weights=metadata["decode"]["shared_weights"],
                  cases=[], teacher_tensors=0, greedy_logit_rows=0, greedy_state_tensors=0,
                  checkpoint=metadata["decode"]["checkpoint"], package_sha256={k: v["package_sha256"] for k, v in metadata.items()},
                  attention_library_sha256=sha256(attention_library),
                  binary_sha256={p.name: sha256(p) for p in (args.binary, args.cli)})
    for case, actual in zip(cases, native["cases"]):
        name = case["name"]
        ref = expected[name]
        assert actual["name"] == name and actual["position"] == ref["position"]
        lengths = ([len(case["prompt"])] + [1] * (len(ref["ids"]) - 1) if "prompt" in case
                   else list(map(len, case["steps"])))
        assert actual["prefill_calls"] == sum(n > 1 for n in lengths)
        assert actual["decode_calls"] == lengths.count(1)
        if "prompt" in case:
            assert actual["ids"] == ref["ids"]
            compare_file(args.output / "native" / f"{name}.f32", ref["rows"])
            compare_file(args.output / "native" / f"{name}.states", ref["state"])
            stop = "eos" if ref["ids"][-1] in case.get("eos", eos) else "limit"
            assert actual["stop"] == stop
            report["greedy_logit_rows"] += len(ref["rows"])
            report["greedy_state_tensors"] += len(ref["state"])
            actual["text"] = tokenizer.decode(ref["ids"], skip_special_tokens=True)
        else:
            compare_file(args.output / "native" / f"{name}.states", (t for row in ref["rows"] for t in row))
            assert actual["kv_inputs_updated_in_place"] and actual["non_kv_inputs_unchanged"]
            assert actual["kv_write_regions_preserved"]
            report["teacher_tensors"] += sum(len(row) for row in ref["rows"])
            if case.get("check_overflow"):
                assert actual["overflow_rejected"]
        report["cases"].append(dict(actual, bitwise_equal=True))
    for first, repeat, suffix in (("prefill_decode", "repeat", "states"), ("capital", "capital_repeat", "f32"),
                                   ("capital", "capital_repeat", "states"),
                                   ("mixed_dispatch", "mixed_dispatch_repeat", "states")):
        assert sha256(args.output / "native" / f"{first}.{suffix}") == sha256(args.output / "native" / f"{repeat}.{suffix}")
    prompt_path, ids_path, trace_path = [args.output / n for n in ("prompt.ids", "generated.ids", "cli.f32")]
    prompt_path.write_text(" ".join(map(str, capital["prompt"])) + "\n")
    run([args.cli.resolve(), args.packages.resolve(), prompt_path.resolve(), ids_path.resolve(),
         "--max-new-tokens", capital["limit"], "--trace", trace_path.resolve()], env, args.output / "cli.log")
    assert [int(x) for x in ids_path.read_text().split()] == expected["capital"]["ids"]
    compare_file(trace_path, expected["capital"]["rows"])
    for binary in (args.binary, args.cli):
        linked = subprocess.check_output(["ldd", str(binary.resolve())], env=env, text=True)
        (args.output / f"{binary.name}.ldd").write_text(linked)
        assert "libpython" not in linked and "libtorch_python" not in linked
    print("Input/package rejection checks", flush=True)
    report["rejections"] = reject_invalid(args, env, metadata)
    report.update(native_runtime_passed=True, no_python_library=True, repeated_requests_bitwise=True, cli_bitwise=True,
                  dispatch_counts_checked=True)
    root = ROOT
    sources = [Path(__file__).relative_to(root), Path("compiler/qwen35/config.py"), Path("compiler/qwen35/prepare.py"),
               Path("runtime/qwen35/runtime.h"), Path("runtime/main.cpp"),
               Path("runtime/validate.cpp")]
    sources += ["compiler/common/artifacts.py", "compiler/common/metadata.py", "compiler/qwen35/packages.py", "compiler/common/tests/validation.py", "compiler/common/weights.py", "compiler/qwen35/reference.py", "compiler/qwen35/tests/common.py", "runtime/common/io.h", "runtime/common/generation.h", "runtime/common/weights.h", "runtime/common/sha256.h"]
    sources += list(NATIVE_RUNTIME_SOURCES)
    sources += ["compiler/common/attention.py", "compiler/qwen35/adapters/attention.py",
                "runtime/common/attention.cpp", "runtime/common/attention.h"]
    sources = list(dict.fromkeys(sources))
    report["source_sha256"] = {str(p): sha256(root / p) for p in sources}
    (args.output / "summary.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({k: report[k] for k in ("native_runtime_passed", "teacher_tensors", "greedy_logit_rows",
                                           "greedy_state_tensors", "hf_numerical_acceptance")}, indent=2))


if __name__ == "__main__":
    main()
