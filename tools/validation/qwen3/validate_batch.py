"""Validate one dynamic-batch Qwen3 bundle at B=1..8 through HF, eager, AOTI and C++."""

import argparse
import gc
import hashlib
import json
from pathlib import Path
import subprocess

import torch
from transformers import DynamicCache

from compiler.common.artifacts import read_json, require, write_json
from compiler.common.metadata import read_package_metadata
from compiler.qwen3.adapters.qwen3_flash import FlashStep
from compiler.qwen3.model import load_model
from tools.prepare.common import encode_prompt, load_tokenizer
from tools.validation.common.attention import load_packaged_attention_ops
from tools.validation.common.runner import ExplicitRuntime, capture_state, load_aoti_runtime
from tools.validation.common.validation import (
    bits_equal,
    tensor_metrics,
    record_failure,
    raw_bytes,
)
from tools.common import native_environment


def case_groups(tokenizer, capacity):
    texts = [
        "What is the capital of France?",
        "Explain how photosynthesis helps plants grow, in one short sentence.",
        "Name the largest planet.",
        "How does a bicycle stay balanced while moving?",
        "List two colors found in a rainbow.",
        "Describe why water freezes when it becomes cold.",
        "What is seven times eight?",
        "Give one practical way to reduce household electricity consumption.",
    ]
    prompts = [encode_prompt(tokenizer, text) for text in texts]
    require(len(prompts[0]) != len(prompts[1]), "Unexpected prompt lengths")
    continuation = [
        tokenizer.encode(text, add_special_tokens=False)[:3]
        for text in ["Paris is beautiful.", "Rome is beautiful."] * 4
    ]
    require(all(len(row) == 3 for row in continuation), "Expected three teacher tokens")
    groups = {}
    for batch_size in range(1, 9):
        steps = [prompts[:batch_size]] + [
            [[row[i]] for row in continuation[:batch_size]] for i in range(3)
        ]
        groups[f"b{batch_size}"] = (
            texts[:batch_size],
            common_cases(steps, capacity) + multi_row_cases(steps, capacity),
        )
    groups["transitions"] = (texts, batch_transition_cases(prompts, continuation))
    for _, cases in groups.values():
        for case in cases:
            case["state_steps"] = [len(case["batch_steps"]) - 1]
    return groups


def common_cases(steps, capacity):
    prompts = steps[0]
    cases = [
        # Baseline: real prompts followed by three teacher-forced decode steps.
        dict(name="prompts", batch_steps=steps),
        # Repeat after reset; logits and populated KV must match the baseline exactly.
        dict(name="reset", batch_steps=steps),
        # Fill KV with 17; unwritten slots must not affect logits or populated KV.
        dict(name="poison", batch_steps=steps, fill=17),
    ]
    # 1/2: decode/minimum prefill; 513: cross a chunk boundary with a one-token tail.
    # Full capacity: reject the next call without changing cache or positions.
    for length in sorted({1, 2, min(513, capacity), capacity}):
        rows = [(row * (length // len(row) + 1))[:length] for row in prompts]
        chunks = [[row[start : start + 512] for row in rows] for start in range(0, length, 512)]
        if length < capacity:
            chunks += steps[1 : 1 + min(3, capacity - length)]
        cases.append(
            dict(name=f"length_{length}", batch_steps=chunks, check_overflow=length == capacity)
        )
    # Padding token IDs must not affect valid logits or populated KV.
    cases.append(dict(name="padding", batch_steps=steps, padding_token=17))
    return cases


def multi_row_cases(steps, capacity):
    prompts = steps[0]
    batch_size = len(prompts)
    if batch_size == 1:
        return []
    cases = [
        # Change row zero only; all other rows' logits and KV must stay identical.
        dict(name="perturb", batch_steps=[[list(reversed(prompts[0])), *prompts[1:]]] + steps[1:])
    ]
    for pattern in (
        (1, 2),  # One real token in a padded two-token prefill.
        (511, 512),  # Unequal lengths within one full prefill call.
        (512, 513),  # Only the longer row consumes a second chunk.
        (19, 513),  # Short row waits through a second chunk, then resumes decode.
        (513, 19),  # Repeat with the long row first.
        (19, min(900, capacity - 3)),  # Large length gap with room for three decode steps.
        (min(900, capacity - 3), 19),  # Repeat with the long row first.
        (capacity, capacity - 7),  # A full row rejects overflow despite space in another row.
        (capacity - 7, capacity),  # Repeat with the second row full.
    ):
        lengths = tuple(pattern[row % len(pattern)] for row in range(batch_size))
        if max(lengths) > capacity:
            continue
        rows = [(row * (length // len(row) + 1))[:length] for row, length in zip(prompts, lengths)]
        chunks = [
            [row[start : start + 512] for row in rows] for start in range(0, max(lengths), 512)
        ]
        if max(lengths) + 3 <= capacity:
            chunks += steps[1:]
        cases.append(
            dict(
                name=f"lengths_{lengths[0]}_{lengths[1]}",
                batch_steps=chunks,
                check_overflow=max(lengths) == capacity,
            )
        )
    if batch_size > 2:
        # Advance one row at a time from distinct positions; idle rows retain their state.
        rows = [
            (prompt * (600 // len(prompt) + 1))[: min(19 + 73 * row, capacity - 1)]
            for row, prompt in enumerate(prompts)
        ]
        chunks = [
            [row[start : start + 512] for row in rows]
            for start in range(0, max(map(len, rows)), 512)
        ]
        chunks += [
            [steps[1][row] if row == active else [] for row in range(batch_size)]
            for active in range(batch_size)
        ]
        cases.append(dict(name="staggered_resume", batch_steps=chunks))
    return cases


def batch_transition_cases(prompts, continuation):
    return [
        # 8->1->3->5->8: omitted rows preserve their cache and resume when restored.
        dict(
            name="changing_batch",
            batch_steps=[
                prompts,
                [[continuation[0][0]]],
                [row[:2] for row in continuation[:3]],
                [[row[0]] for row in continuation[:5]],
                [[row[1]] for row in continuation],
            ],
        ),
        # 1->3->8: new rows start fresh while existing rows continue from their caches.
        dict(
            name="growing_batch",
            batch_steps=[
                prompts[:1],
                [[continuation[0][0]], *prompts[1:3]],
                [[row[1]] for row in continuation[:3]] + prompts[3:],
            ],
        ),
    ]


def collect_hf(model, steps):
    caches = [DynamicCache() for _ in range(max(map(len, steps)))]
    logits = []
    for rows in steps:
        scores = []
        for row, cache in zip(rows, caches):
            if not row:
                scores.append(torch.zeros((1, model.config.vocab_size), dtype=torch.bfloat16))
                continue
            position = cache.get_seq_length()
            score = model(
                input_ids=torch.tensor([row], device="cuda"),
                past_key_values=cache,
                cache_position=torch.arange(position, position + len(row), device="cuda"),
                use_cache=True,
                logits_to_keep=1,
            ).logits[:, -1]
            require(bool(torch.isfinite(score).all()), "Nonfinite HF logits")
            scores.append(score.cpu())
        logits.append(torch.cat(scores))
    state = [
        [getattr(cache, kind)[layer].cpu() for cache in caches]
        for layer in range(model.config.num_hidden_layers)
        for kind in ("key_cache", "value_cache")
    ]
    return logits, [row for tensor in state for row in tensor]


def collect_explicit(runtime, case):
    runtime.reset(case.get("fill", 0))
    logits = [
        runtime.step_batch(rows, padding_token=case.get("padding_token", 0)).cpu()
        for rows in case["batch_steps"]
    ]
    state = [
        tensor[row : row + 1, :, :position].cpu()
        for tensor in runtime.state
        for row, position in enumerate(runtime.positions[: max(map(len, case["batch_steps"]))])
    ]
    return logits, state


def compare_stage(report, name, stage, actual, reference, *, logits):
    for index, (a, b) in enumerate(zip(actual, reference)):
        metric = tensor_metrics(
            a,
            b,
            logits=logits,
            rows=(a.float(), b.float()) if logits else (a.float().flatten(1), b.float().flatten(1)),
        )
        context = dict(case=name, stage=stage, kind="logits" if logits else "state", index=index)
        report["comparisons"].append(dict(**context, required=logits, **metric))
        record_failure(report["failures"], metric, required=logits, **context)


def write_trace(path, tensors):
    """Preserve every byte, storing all-zero chunks as sparse file holes."""
    digest = hashlib.sha256()
    with path.open("wb") as stream:
        for tensor in tensors:
            data = raw_bytes(tensor)
            digest.update(data)
            for offset in range(0, len(data), 65536):
                chunk = data[offset : offset + 65536]
                if chunk.count(0) == len(chunk):
                    stream.seek(len(chunk), 1)
                else:
                    stream.write(chunk)
        stream.truncate()
    return digest.hexdigest()


def validate_python_group(args, group, texts, cases, meta, hf, eager, aoti):
    output = args.output / group
    output.mkdir()
    batch_size = len(texts)
    manifest = dict(trace_format=2, cases=cases)
    write_json(output / "cases.json", manifest)
    report = dict(
        batch_size=batch_size,
        prompts=texts,
        prompt_tokens=[len(row) for row in cases[0]["batch_steps"][0]],
        runtime_abi_version=meta["runtime_abi_version"],
        numerical_policy=dict(name="shared_bf16", version=1),
        comparisons=[],
        failures=[],
        structural_checks=[],
        parity_digests={},
        cases=[],
    )
    base_logits = None
    for case in cases:
        name = case["name"]
        print(f"{group}: validating {name}", flush=True)
        hf_logits, hf_state = collect_hf(hf, case["batch_steps"])
        eager_logits, eager_state = collect_explicit(eager, case)
        actual_logits, actual_state = collect_explicit(aoti, case)
        stages = [
            ("eager/hf", (eager_logits, eager_state), (hf_logits, hf_state)),
            ("aoti/eager", (actual_logits, actual_state), (eager_logits, eager_state)),
            ("aoti/hf", (actual_logits, actual_state), (hf_logits, hf_state)),
        ]
        for stage, actual, reference in stages:
            compare_stage(report, name, stage, actual[0], reference[0], logits=True)
            compare_stage(report, name, stage, actual[1], reference[1], logits=False)
        if name == "prompts":
            base_logits = actual_logits
            base_row_state = [tensor.clone() for tensor in actual_state]
        elif name in ("reset", "poison", "perturb", "padding"):
            for current, base in zip(actual_logits, base_logits):
                require(
                    bits_equal(
                        current[1:] if name == "perturb" else current,
                        base[1:] if name == "perturb" else base,
                    ),
                    f"{name} changed outputs",
                )
            for index, (current, base) in enumerate(zip(actual_state, base_row_state)):
                if name != "perturb" or index % batch_size != 0:
                    require(
                        bits_equal(current, base),
                        f"{name} changed unaffected KV row {index % batch_size}",
                    )
            report["structural_checks"].append(name)
        prefix = args.output / "expected" / f"{group}_{name}"
        report["parity_digests"][name] = {
            ".states": write_trace(
                prefix.with_suffix(f".{case['state_steps'][0]}.states"),
                capture_state(aoti, meta["state"], case.get("fill", 0)),
            ),
            ".f32": write_trace(prefix.with_suffix(".f32"), (t.float() for t in actual_logits)),
        }
        if case.get("check_overflow"):
            before = [tensor.clone() for tensor in aoti.state]
            positions = list(aoti.positions)
            try:
                aoti.step_batch([[1] for _ in range(batch_size)])
            except ValueError as error:
                require("capacity" in str(error), str(error))
            else:
                raise ValueError("Overflow accepted")
            require(aoti.positions == positions, "Overflow changed position")
            require(
                all(bits_equal(a, b) for a, b in zip(aoti.state, before)), "Overflow mutated state"
            )
            report["structural_checks"].append("overflow")
            del before
        report["cases"].append(
            dict(
                name=name,
                position=aoti.position,
                positions=list(aoti.positions),
                steps=len(actual_logits),
            )
        )
        write_json(output / "report.json", report)


@torch.inference_mode()
def validate_python(args, meta):
    load_packaged_attention_ops(args.packages)
    tokenizer = load_tokenizer(args.model_dir)
    groups = case_groups(tokenizer, meta["cache_length"])
    hf = load_model(args.model_dir)
    adapted = FlashStep(load_model(args.model_dir))
    eager = ExplicitRuntime(
        dict(prefill=adapted, decode=adapted),
        meta["state"],
        meta["vocab_size"],
        meta["cache_length"],
        logits_only=True,
        require_write=True,
    )
    aoti = load_aoti_runtime(args.packages, model_type="qwen3", specs=meta["state"])
    try:
        for group, (texts, cases) in groups.items():
            validate_python_group(args, group, texts, cases, meta, hf, eager, aoti)
    finally:
        eager.close()
        aoti.close()
    return list(groups)


def validate_native(args, groups):
    cases = []
    for group in groups:
        manifest = read_json(args.output / group / "cases.json")
        cases.extend(dict(case, name=f"{group}_{case['name']}") for case in manifest["cases"])
    write_json(
        args.output / "cases.json",
        dict(
            trace_format=2,
            cases=cases,
            expected_directory=str((args.output / "expected").resolve()),
        ),
    )
    subprocess.run(
        [
            str(args.native.resolve()),
            str(args.packages.resolve()),
            str((args.output / "cases.json").resolve()),
            str((args.output / "native").resolve()),
        ],
        env=native_environment(),
        check=True,
    )
    native = read_json(args.output / "native/native.json")
    require(
        native["loader_pairs"] == 1 and native["no_python_library"] and native["attention_library"],
        "Invalid native execution environment",
    )
    require(
        [row["name"] for row in native["cases"]] == [case["name"] for case in cases],
        "Native case coverage differs",
    )
    rows = {row["name"]: row for row in native["cases"]}
    results = {}
    for group in groups:
        output = args.output / group
        report = read_json(output / "report.json")
        selected = []
        for expected in report["cases"]:
            row = rows[f"{group}_{expected['name']}"]
            require(row["positions"] == expected["positions"], "Native per-row positions differ")
            require(row["bitwise_parity"], "Native parity check missing")
            selected.append(dict(row, name=expected["name"]))
        report["native"] = dict(native, cases=selected)
        report["native_bitwise_parity"] = True
        write_json(output / "report.json", report)
        results[group] = dict(required_failures=len(report["failures"]))
    write_json(args.output / "summary.json", results)
    print(json.dumps(results))
    return results


def validate(args):
    require(__debug__, "Validation requires assertions enabled")
    require(not args.output.exists(), "Use a fresh validation output directory")
    meta = read_package_metadata(args.packages, "qwen3")["decode"]
    require(
        meta["checkpoint"]["manifest"] == read_json(args.model_dir / "model_manifest.json"),
        "Checkpoint manifest differs",
    )
    (args.output / "expected").mkdir(parents=True)
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    groups = validate_python(args, meta)
    # Native validation starts after the Python model set has been released.
    gc.collect()
    torch.cuda.empty_cache()
    return validate_native(args, groups)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--packages", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--native", type=Path, default=Path("build/qwen3/xinfer_validate"))
    args = parser.parse_args()
    results = validate(args)
    raise SystemExit(any(item["required_failures"] for item in results.values()))


if __name__ == "__main__":
    main()
