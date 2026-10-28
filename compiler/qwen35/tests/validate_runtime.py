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

from compiler.qwen35.config import ARTIFACT_DIR, CACHE_LENGTH, PREFILL, TEXT_DIR
from compiler.common.artifacts import sha256, ROOT
from compiler.qwen35.prepare import encode_prompt
from compiler.qwen35.model import configure_text
from compiler.qwen35.tests.common import fixtures
from compiler.qwen35.reference import PythonRuntime
from compiler.common.validation import cpu_copy, compare_file
from compiler.common.artifacts import run


def reject_invalid(args, env, metadata):
    directory = args.output / "rejections"
    directory.mkdir()
    cases = [("empty", "", "1", "Expected 1..512"),
             ("oversized", "42 " * 513, "1", "Expected 1..512"),
             ("negative_id", "-1", "1", "Token ID out of range"),
             ("large_id", str(metadata["decode"]["vocabulary"]), "1", "Token ID out of range"),
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
        ("mixed_checkpoint", lambda m: m["prefill"]["checkpoint"].update(revision="different"), "Package metadata differs: checkpoint"),
        ("mixed_state", lambda m: m["prefill"]["state"][0]["shape"].__setitem__(-1, 5), "Package metadata differs: state"),
        ("mixed_numerics", lambda m: m["prefill"].update(eager_numerics=not m["decode"].get("eager_numerics", False)), "Package numerical settings differ"),
        ("entry_point", lambda m: m["prefill"].update(tokens=63),
         "Expected dynamic T=2..512/T=1" if metadata["prefill"].get("dynamic_prefill", False) else "Expected T=64/T=1"),
        ("torch_version", lambda m: [v.update(torch="0.0.0") for v in m.values()], "Torch version mismatch"),
        ("state_dtype", lambda m: [v["state"][0].update(dtype="float32") for v in m.values()], "Invalid state order or dtype"),
        ("abi_order", lambda m: [v["input_order"].reverse() for v in m.values()], "Invalid tensor ABI order"),
        ("gpu_arch", lambda m: [v.update(sm=[0, 0]) for v in m.values()], "GPU architecture mismatch"),
    ]
    if metadata["prefill"].get("dynamic_prefill", False):
        mutations += [
            ("prefill_range", lambda m: m["prefill"].update(token_range=[2, 513]), "Expected dynamic T=2..512/T=1"),
            ("mixed_mode", lambda m: m["prefill"].update(dynamic_prefill=False), "Package prefill modes differ"),
            ("mixed_library", lambda m: m["prefill"]["delta_library"].update(sha256="different"), "Package DeltaNet libraries differ"),
            ("library_size", lambda m: [v["delta_library"].update(bytes=0) for v in m.values()], "DeltaNet library size mismatch"),
        ]
    for name, mutate, error in mutations:
        packages = directory / name
        packages.mkdir()
        modified = copy.deepcopy(metadata)
        mutate(modified)
        if (args.packages / "libxinfer_delta_ops.so").exists():
            (packages / "libxinfer_delta_ops.so").symlink_to((args.packages / "libxinfer_delta_ops.so").resolve())
        for kind, meta in modified.items():
            (packages / f"{kind}.json").write_text(json.dumps(meta))
            (packages / f"{kind}.pt2").symlink_to((args.packages / f"{kind}.pt2").resolve())
        output = packages / "generated.ids"
        run([args.cli, packages, prompt, output], env, directory / f"{name}.log", success=False, error=error)
        assert not output.exists()
        passed.append(name)
    return passed


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=TEXT_DIR)
    parser.add_argument("--packages", type=Path, default=ARTIFACT_DIR / "text")
    parser.add_argument("--binary", type=Path, default=Path("work/qwen35/build/qwen35_validate"))
    parser.add_argument("--cli", type=Path, default=Path("work/qwen35/build/qwen35_aoti"))
    parser.add_argument("--output", type=Path, default=ARTIFACT_DIR / "runtime-validation")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    configure_text()
    metadata = {k: json.loads((args.packages / f"{k}.json").read_text()) for k in ("prefill", "decode")}
    for kind, meta in metadata.items():
        assert meta["package_sha256"] == sha256(args.packages / f"{kind}.pt2")
        assert meta["checkpoint"] == json.loads((args.model_dir / "manifest.json").read_text())
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, local_files_only=True)
    dynamic_prefill = metadata["prefill"].get("dynamic_prefill", False)
    cases = fixtures(tokenizer, dynamic_prefill)
    seed = cases[1]["steps"][0]
    for length in (2, 63, 65, 129, 511):
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
    eos = metadata["decode"]["generation_config"]["eos_token_id"]
    eos = eos if isinstance(eos, list) else [eos]
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
    assert native["loader_pairs"] == 1 and native["no_python_library"] and len(native["cases"]) == len(cases)
    report = dict(comparison="C++ vs Python AOTI", hf_numerical_acceptance="open: Q35-NUM-001 and step-3 state failures",
                  cases=[], teacher_tensors=0, greedy_logit_rows=0, greedy_state_tensors=0,
                  checkpoint=metadata["decode"]["checkpoint"], package_sha256={k: v["package_sha256"] for k, v in metadata.items()},
                  binary_sha256={p.name: sha256(p) for p in (args.binary, args.cli)})
    for case, actual in zip(cases, native["cases"]):
        name = case["name"]
        ref = expected[name]
        assert actual["name"] == name and actual["position"] == ref["position"]
        if dynamic_prefill:
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
            assert actual["inputs_unchanged"]
            report["teacher_tensors"] += sum(len(row) for row in ref["rows"])
            if case.get("check_overflow"):
                assert actual["overflow_rejected"]
        report["cases"].append(dict(actual, bitwise_equal=True))
    for first, repeat, suffix in (("prefill_decode", "repeat", "states"), ("capital", "capital_repeat", "f32"),
                                   ("capital", "capital_repeat", "states")):
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
                  dynamic_prefill=dynamic_prefill, dispatch_counts_checked=dynamic_prefill)
    root = ROOT
    sources = [Path(__file__).relative_to(root), Path("compiler/qwen35/config.py"), Path("compiler/qwen35/prepare.py"),
               Path("runtime/qwen35/runtime.h"), Path("runtime/qwen35/main.cpp"),
               Path("runtime/qwen35/tests/text_validate.cpp")]
    sources += ["compiler/common/artifacts.py", "compiler/common/validation.py", "compiler/qwen35/reference.py", "compiler/qwen35/tests/common.py", "runtime/common/io.h", "runtime/common/generation.h"]
    report["source_sha256"] = {str(p): sha256(root / p) for p in sources}
    (args.output / "summary.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({k: report[k] for k in ("native_runtime_passed", "teacher_tensors", "greedy_logit_rows",
                                           "greedy_state_tensors", "hf_numerical_acceptance")}, indent=2))


if __name__ == "__main__":
    main()
