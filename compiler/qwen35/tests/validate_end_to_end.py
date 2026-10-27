"""Compare HF full-prompt/chunked eager, Python AOTI and native text generation."""

import argparse
from collections import Counter
import gc
import json
import os
from pathlib import Path
import shutil

import torch
from transformers import AutoTokenizer

from compiler.qwen35.config import ARTIFACT_DIR, TEXT_DIR
from compiler.common.artifacts import sha256, ROOT
from compiler.qwen35.prepare import encode_prompt
from compiler.qwen35.reference import HFRuntime
from compiler.qwen35.model import configure_text, load_model
from compiler.common.validation import bits_equal
from compiler.qwen35.reference import PythonRuntime
from compiler.common.validation import compare_file, cpu_copy
from compiler.common.artifacts import run
from compiler.qwen35.tests.common import compare


def greedy(runtime, case, eos):
    runtime.reset()
    logits = runtime.step(case["prompt"])
    ids, rows = [], []
    for i in range(case["limit"]):
        rows.append(logits.cpu().float())
        token = logits.argmax(-1).item()
        ids.append(token)
        if token in eos or i + 1 == case["limit"]:
            break
        logits = runtime.step([token])
    return dict(ids=ids, rows=rows, state=cpu_copy(runtime.state), position=runtime.position)


def trace(runtime, cases, specs):
    values = {}
    for case in cases:
        poison = case.get("poison_kv", 0)
        if isinstance(runtime, HFRuntime):
            runtime.reset(poison)
        else:
            runtime.reset()
            for tensor, spec in zip(runtime.state, specs):
                if spec["kind"] in ("key", "value"):
                    tensor.fill_(poison)
        position = 0
        for i, tokens in enumerate(case["steps"]):
            logits = runtime.step(tokens)
            row = cpu_copy((logits, *runtime.state))
            values[(case["name"], i)] = row
            position += len(tokens)
            assert runtime.position == position
            if "same_as" in case:
                clean = values[(case["same_as"], i)]
                assert bits_equal(row[0], clean[0]), "Masked KV changes logits"
                for a, b, spec in zip(row[1:], clean[1:], specs):
                    if spec["kind"] in ("key", "value"):
                        assert bits_equal(a[:, :, :position], b[:, :, :position]), "Masked KV changes valid state"
                        assert (a[:, :, position:] == poison).all(), "Future KV overwritten"
                    else:
                        assert bits_equal(a, b), "Masked KV changes DeltaNet state"
        print(case["name"], "position", position, flush=True)
    return values


def token_difference(actual, expected):
    i = next((i for i, (a, b) in enumerate(zip(actual["ids"], expected["ids"])) if a != b),
             min(len(actual["ids"]), len(expected["ids"])))
    result = dict(first_difference=i, actual_length=len(actual["ids"]), expected_length=len(expected["ids"]))
    if i < min(len(actual["ids"]), len(expected["ids"])):
        scores = expected["rows"][i][0]
        top = scores.topk(2).values
        result.update(actual_id=actual["ids"][i], expected_id=expected["ids"][i],
                      reference_top2_margin=(top[0] - top[1]).item(),
                      reference_gap_to_actual=(top[0] - scores[actual["ids"][i]]).item())
    return result


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=TEXT_DIR)
    parser.add_argument("--packages", type=Path, default=ARTIFACT_DIR / "text")
    parser.add_argument("--native", type=Path, default=Path("work/qwen35/build/qwen35_validate"))
    parser.add_argument("--output", type=Path, default=ARTIFACT_DIR / "end-to-end")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    configure_text()
    metadata = {k: json.loads((args.packages / f"{k}.json").read_text()) for k in ("prefill", "decode")}
    for kind, meta in metadata.items():
        assert meta["torch"] == torch.__version__ and meta["package_sha256"] == sha256(args.packages / f"{kind}.pt2")
        assert meta["checkpoint"] == json.loads((args.model_dir / "manifest.json").read_text())
    specs = metadata["decode"]["state"]
    eos = metadata["decode"]["generation_config"]["eos_token_id"]
    eos = eos if isinstance(eos, list) else [eos]
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, local_files_only=True)
    chats = [dict(name=name, prompt=encode_prompt(tokenizer, text), text=text, limit=64) for name, text in (
        ("capital", "What is the capital of France? Answer briefly."),
        ("arithmetic", "What is 17 + 25? Answer with only the number."),
        ("chinese", "用一句话解释什么是编译器。"))]
    paragraph = tokenizer.encode("The capital of France is Paris. A small compiler can run a language model. ")
    stream = (paragraph * (1024 // len(paragraph) + 1))[:1024]
    cases = []
    for length in (1, 2, 63, 64, 65, 127, 128, 129, 511, 512):
        decode = 64 if length == 512 else 8
        cases.append(dict(name=f"length_{length}", steps=[stream[:length]] + [[t] for t in stream[length:length + decode]]))
    cases.append(dict(name="context_end", steps=[stream[:512], stream[512:960], stream[960:1023], stream[1023:]],
                      check_overflow=True))
    for name in ("length_2", "length_65", "context_end"):
        clean = next(c for c in cases if c["name"] == name)
        cases.append(dict(clean, name=name + "_poison", same_as=name, poison_kv=3))
    cases.append(dict(next(c for c in cases if c["name"] == "length_65"), name="repeat_65"))
    model = load_model(args.model_dir)
    matched = HFRuntime(model, chunked=True)
    print("HF greedy reference with matched partitions", flush=True)
    greedy_hf = {c["name"]: greedy(matched, c, eos) for c in chats}
    for c in chats:
        cases.append(dict(name=c["name"] + "_teacher", steps=[c["prompt"]] + [[t] for t in greedy_hf[c["name"]]["ids"][:-1]]))
    print("HF teacher-forced, matched partitions", flush=True)
    reference = trace(matched, cases, specs)
    whole = HFRuntime(model, chunked=False)
    print("HF teacher-forced, full-prompt calls", flush=True)
    unpartitioned = trace(whole, cases, specs)
    greedy_whole = {c["name"]: greedy(whole, c, eos) for c in chats}
    del matched, whole, model
    gc.collect()
    torch.cuda.empty_cache()
    print("Python AOTI", flush=True)
    compiled = PythonRuntime(args.packages, specs)
    aoti = trace(compiled, cases, specs)
    greedy_aoti = {c["name"]: greedy(compiled, c, eos) for c in chats}
    del compiled
    gc.collect()
    torch.cuda.empty_cache()
    print("Independent FP32 controls", flush=True)
    fp32_cases = [c for c in cases if c["name"] in ("length_512", "context_end", "capital_teacher")]
    fp32_model = load_model(args.model_dir, dtype=torch.float32)
    fp32_runner = HFRuntime(fp32_model, chunked=True)
    fp32 = trace(fp32_runner, fp32_cases, specs)
    del fp32_runner, fp32_model
    gc.collect()
    torch.cuda.empty_cache()
    manifest = args.output / "cases.json"
    manifest.write_text(json.dumps(dict(cases=cases + chats), indent=2, ensure_ascii=False) + "\n")
    env = {k: v for k, v in os.environ.items() if k not in ("VIRTUAL_ENV", "PYTHONPATH", "PYTHONHOME", "LD_LIBRARY_PATH")}
    env.update(PATH="/usr/bin:/bin", PYTHONHOME="/nonexistent-python-home")
    print("Fresh native process", flush=True)
    run([args.native.resolve(), args.packages.resolve(), manifest.resolve(), (args.output / "native").resolve()],
        env, args.output / "native.log")
    native = json.loads((args.output / "native/native.json").read_text())
    assert native["no_python_library"] and native["loader_pairs"] == 1 and len(native["cases"]) == len(cases + chats)
    for case, actual in zip(cases + chats, native["cases"]):
        name = case["name"]
        assert actual["name"] == name
        if "steps" in case:
            compare_file(args.output / "native" / f"{name}.states", (t for i in range(len(case["steps"])) for t in aoti[(name, i)]))
            assert actual["position"] == sum(map(len, case["steps"])) and actual["inputs_unchanged"]
            if case.get("check_overflow"):
                assert actual["overflow_rejected"]
        else:
            expected = greedy_aoti[name]
            assert actual["ids"] == expected["ids"] and actual["position"] == expected["position"]
            compare_file(args.output / "native" / f"{name}.f32", expected["rows"])
            compare_file(args.output / "native" / f"{name}.states", expected["state"])
    for i in range(9):
        assert all(bits_equal(a, b) for a, b in zip(aoti[("length_65", i)], aoti[("repeat_65", i)]))
    report = dict(limit=0.05, comparisons=[], failures=[], top1_differences=[], greedy=[], greedy_failures=[],
                  checkpoint=metadata["decode"]["checkpoint"], package_sha256={k: v["package_sha256"] for k, v in metadata.items()},
                  native_sha256=sha256(args.native), native_bitwise=True, teacher_steps=len(aoti),
                  teacher_tensors=sum(map(len, aoti.values())), no_python_library=True,
                  masked_future_checks=True, repeated_request_bitwise=True, context_overflow_rejected=True)
    for name, actual, expected, enforce in (
        ("aoti_vs_hf_chunked", aoti, reference, True),
        ("aoti_vs_hf_full_prompt", aoti, unpartitioned, True),
        ("hf_chunked_vs_full_prompt", reference, unpartitioned, False),
        ("aoti_vs_fp32", {k: aoti[k] for k in fp32}, fp32, False),
        ("hf_vs_fp32", {k: reference[k] for k in fp32}, fp32, False),
    ):
        compare(report, name, actual, expected, specs, enforce)
    for case in chats:
        name = case["name"]
        actual = greedy_aoti[name]
        row = dict(name=name, prompt=case["text"], aoti_ids=actual["ids"], aoti_text=tokenizer.decode(actual["ids"], skip_special_tokens=True))
        for label, refs in (("hf_chunked", greedy_hf), ("hf_full_prompt", greedy_whole)):
            expected = refs[name]
            row[label] = dict(ids=expected["ids"], text=tokenizer.decode(expected["ids"], skip_special_tokens=True),
                              equal=actual["ids"] == expected["ids"])
            if not row[label]["equal"]:
                report["greedy_failures"].append(dict(case=name, comparison="aoti_vs_" + label, **token_difference(actual, expected)))
        report["greedy"].append(row)
    report["failure_counts"] = dict(Counter(f["comparison"] + "/" + f["tensor"].split(".")[-1] for f in report["failures"]))
    report["passed"] = not (report["failures"] or report["greedy_failures"] or
                             any(r["comparison"].startswith("aoti_vs_hf") for r in report["top1_differences"]))
    root = ROOT
    names = [n for n in ("compiler/qwen35/tests/validate_end_to_end.py", "compiler/qwen35/reference.py",
                      "compiler/qwen35/tests/common.py", "compiler/qwen35/model.py", "compiler/qwen35/config.py", "compiler/qwen35/prepare.py")]
    names += ["runtime/qwen35/runtime.h", "runtime/qwen35/tests/text_validate.cpp"]
    names += ["compiler/common/artifacts.py", "compiler/common/validation.py", "runtime/common/io.h"]
    for name in names:
        target = args.output / "source" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(root / name, target)
    report["source_sha256"] = {n: sha256(root / n) for n in names}
    torch.save({label: values[("context_end", 3)] for label, values in
                (("hf", reference), ("hf_full_prompt", unpartitioned), ("aoti", aoti), ("fp32", fp32))}, args.output / "context_end.pt")
    (args.output / "summary.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({k: report[k] for k in ("passed", "native_bitwise", "teacher_steps", "teacher_tensors",
                                           "failure_counts", "greedy_failures")}, indent=2))
    raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
