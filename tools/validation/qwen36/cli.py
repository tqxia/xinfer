"""Check the production C++ CLI against saved Python AOTI outputs; controller is CPU-only."""

import argparse
import copy
import os
from pathlib import Path
import re
import shutil
import subprocess
import traceback

import torch

from compiler.common.artifacts import require, sha256, write_json
from compiler.common.metadata import read_package_metadata
from compiler.qwen36.config import TEXT_DIR
from compiler.qwen36.packages import validate_model_metadata
from tools.validation.common.validation import raw_bytes, read_ids
from tools.validation.qwen36.aoti import reference_info
from tools.validation.qwen36.eager import load_trace


def fixtures(protocol, reference):
    """Use the prompt-end row for chunked prefill, never a later teacher decode."""
    result = []
    for name in ("capital", "limit_one", "multicall_513"):
        trace, _ = load_trace(reference / "traces" / f"{name}.pt")
        if name == "multicall_513":
            case = next(c for group in protocol["groups"] for c in group if c["name"] == name)
            prompt = [token for chunk in case["steps"][: case["prompt_steps"]] for token in chunk]
            rows = [trace.logits[name, case["prompt_steps"] - 1]]
            ids, limit = [rows[0].argmax(-1).item()], 1
        else:
            case = next(c for c in protocol["chats"] if c["name"] == name)
            prompt, limit = case["prompt"], case["limit"]
            rows, ids = trace.greedy[name]["rows"], trace.greedy[name]["ids"]
        require(len(rows) == len(ids) and 0 < len(ids) <= limit, "Invalid reference generation")
        require(
            all(
                row.device.type == "cpu"
                and row.dtype == torch.float32
                and row.ndim == 2
                and row.shape[0] == 1
                and bool(torch.isfinite(row).all())
                for row in rows
            ),
            "Invalid reference logit rows",
        )
        require(
            all(row.argmax(-1).item() == token for row, token in zip(rows, ids)),
            "Reference IDs disagree with logits",
        )
        eos = protocol["eos_token_ids"]
        require(not any(token in eos for token in ids[:-1]), "Reference continued after EOS")
        stop = "eos" if ids[-1] in eos else "limit"
        require(stop == "eos" or len(ids) == limit, "Incomplete reference generation")
        result.append(
            dict(
                name=name,
                prompt=prompt,
                limit=limit,
                ids=ids,
                data=b"".join(raw_bytes(row) for row in rows),
                position=len(prompt) + len(ids) - 1,
                stop=stop,
            )
        )
    return result


def invoke(args, name, command, env, *, timeout=1200):
    result = subprocess.run(
        [str(item) for item in command],
        cwd=args.output,
        env=env,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    (args.output / f"{name}.stdout.log").write_text(result.stdout)
    (args.output / f"{name}.log").write_text(result.stderr)
    return result


def valid_cases(args, cases, env, report, vocab):
    require(
        all(len(case["data"]) == len(case["ids"]) * vocab * 4 for case in cases),
        "Reference vocabulary differs from package",
    )
    for case in cases:
        name = case["name"]
        prompt, output, trace = [
            args.output / f"{name}.{suffix}" for suffix in ("input.ids", "ids", "f32")
        ]
        prompt.write_text(" ".join(map(str, case["prompt"])) + "\n")
        result = invoke(
            args,
            name,
            [
                args.native,
                args.packages,
                prompt,
                output,
                "--max-output-tokens",
                case["limit"],
                "--trace",
                trace,
            ],
            env | ({"LD_DEBUG": "libs"} if name == "capital" else {}),
        )
        require(result.returncode == 0, f"{name} failed: {result.stderr}")
        require(read_ids(output) == case["ids"], f"{name}: generated IDs differ")
        data = trace.read_bytes()
        require(
            len(data) == len(case["ids"]) * vocab * 4 and data == case["data"],
            f"{name}: raw FP32 logits differ",
        )
        require(
            bool(torch.isfinite(torch.frombuffer(bytearray(data), dtype=torch.float32)).all()),
            f"{name}: nonfinite logits",
        )
        status = re.search(
            r"prompt_tokens=(\d+) generated_tokens=(\d+) consumed_positions=(\d+) stop=(eos|limit)\b",
            result.stderr,
        )
        require(
            status
            and tuple(map(int, status.groups()[:3]))
            == (len(case["prompt"]), len(case["ids"]), case["position"])
            and status[4] == case["stop"],
            f"{name}: incorrect stop/position report",
        )
        if name == "capital":
            require(
                "libpython" not in result.stderr and "libtorch_python" not in result.stderr,
                "Python library loaded",
            )
            require(
                re.search(r"calling init: .*libxinfer_attention_ops\.so", result.stderr),
                "Native attention initialization missing from loader trace",
            )
        report["fixtures"].append(
            dict(
                name=name,
                prompt_tokens=len(case["prompt"]),
                ids=case["ids"],
                position=case["position"],
                stop=case["stop"],
                logit_rows=len(case["ids"]),
                bitwise_equal=True,
            )
        )
        print(f"CLI: {name} bitwise", flush=True)


def rejections(args, env, metadata, report):
    directory = args.output / "rejections"
    directory.mkdir()
    hidden = env | {"CUDA_VISIBLE_DEVICES": ""}
    capacity, vocab = metadata["decode"]["cache_length"], metadata["decode"]["vocab_size"]

    def reject(name, prompt, output, trace, options, error, packages=None):
        before = {
            path: path.read_bytes() if path.exists() else None for path in (prompt, output, trace)
        }
        result = invoke(
            args,
            name,
            [args.native, packages or args.packages, prompt, output, *options],
            hidden,
            timeout=30,
        )
        require(
            result.returncode != 0 and error in result.stderr,
            f"{name}: expected {error!r}, got {result.stderr}",
        )
        require(
            before == {path: path.read_bytes() if path.exists() else None for path in before},
            f"{name}: rejection changed input/output files",
        )
        report["rejections"].append(dict(name=name, cpu_preflight=True, output_unchanged=True))

    cases = [
        ("empty", "", [], f"Expected 1..{capacity} input tokens"),
        ("negative", "-1", [], "Token ID out of range"),
        ("vocab", str(vocab), [], "Token ID out of range"),
        ("fraction", "1.5", [], "Invalid integer"),
        ("long_prompt", "1 " * (capacity + 1), [], f"Expected 1..{capacity} input tokens"),
        ("zero_budget", "1", ["--max-output-tokens", "0"], "Generation budget"),
        ("context_budget", "1", ["--max-output-tokens", str(capacity)], "Generation budget"),
        ("retired_prompt_limit", "1", ["--max-prompt-length", "1"], "Unknown option"),
        ("retired_output_limit", "1", ["--max-new-tokens", "1"], "Unknown option"),
        ("missing_value", "1", ["--trace"], "Missing option value"),
    ]
    for name, text, options, error in cases:
        prompt, output, trace = [
            directory / f"{name}.{suffix}" for suffix in ("input.ids", "ids", "f32")
        ]
        prompt.write_text(text)
        reject(name, prompt, output, trace, ["--max-output-tokens", "1", *options], error)
    prompt, output, trace = [directory / name for name in ("valid.ids", "output.ids", "trace.f32")]
    prompt.write_text("1\n")
    output.write_text("retain existing output\n")
    reject("existing_output", prompt, output, trace, [], "Output already exists")
    output.unlink()
    trace.write_text("retain existing trace\n")
    reject("existing_trace", prompt, output, trace, ["--trace", trace], "Trace path already exists")
    trace.unlink()
    for name, destination, target, error in (
        ("input_output_collision", prompt, None, "Output already exists"),
        ("trace_input_collision", output, prompt, "Trace path must differ"),
        ("trace_output_collision", output, output, "Trace path must differ"),
    ):
        reject(name, prompt, destination, trace, ["--trace", target] if target else [], error)

    clone = directory / "packages"
    clone.mkdir()
    for filename in ("prefill.pt2", "decode.pt2", "prefill.json", "decode.json"):
        shutil.copy2(args.packages / filename, clone / filename)
    archive = metadata["decode"]["shared_weights"]["file"]
    os.link(args.packages / archive, clone / archive)
    require(os.path.samefile(args.packages / archive, clone / archive), "Weight archive was copied")
    for name, error in (
        ("fp8_format", "Unsupported Qwen3.6 FP8 backend"),
        ("checkpoint_revision", "Unsupported Qwen3.6 checkpoint"),
        ("state_geometry", "Unsupported Qwen3.6 state geometry"),
    ):
        bad = copy.deepcopy(metadata)
        for kind, value in bad.items():
            if name == "fp8_format":
                value["model_specific"]["fp8_format"] = "e5m2"
            elif name == "checkpoint_revision":
                value["checkpoint"]["revision"] = "unsupported"
            else:
                value["state"][0]["shape"][1] = 8193
            write_json(clone / f"{kind}.json", value)
        reject(name, prompt, output, trace, [], error, clone)
    report["metadata_probe_archive"] = dict(hardlinked=True, bytes=(clone / archive).stat().st_size)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("native", "packages", "reference", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument(
        "--model-dir", type=Path, default=TEXT_DIR, help="Only model_manifest.json is read"
    )
    args = parser.parse_args(argv)
    for name in ("native", "packages", "reference", "model_dir", "output"):
        setattr(args, name, getattr(args, name).resolve())
    args.output.mkdir(parents=True, exist_ok=False)
    report = dict(
        model_type="qwen36",
        passed=False,
        cli_passed=False,
        model_accepted=False,
        fixtures=[],
        rejections=[],
        scope="Production CLI parity and CPU preflight; no accuracy or performance acceptance",
    )
    try:
        require(not torch.cuda.is_initialized(), "Controller must remain CPU-only")
        pair = read_package_metadata(args.packages, model_type="qwen36")
        for metadata in pair.values():
            validate_model_metadata(metadata)
        protocol, source, hf = reference_info(args.reference, "native", pair, args.model_dir)
        write_json(args.output / "protocol.json", protocol)
        report.update(
            hf_numerical_reference=hf,
            compiler_eager_reference=dict(
                passed=source["functional_passed"],
                failures=len(source["failures"]),
                summary_sha256=sha256(args.reference / "summary.json"),
            ),
            package_sha256=source["package_sha256"],
            native_sha256=sha256(args.native),
        )
        removed = (
            "VIRTUAL_ENV PYTHONPATH PYTHONHOME LD_LIBRARY_PATH LD_DEBUG LD_DEBUG_OUTPUT".split()
        )
        env = {key: value for key, value in os.environ.items() if key not in removed}
        env.update(PATH="/usr/bin:/bin", PYTHONHOME="/nonexistent-python-home")
        linked = invoke(args, "ldd", ["/usr/bin/ldd", args.native], env)
        require(
            linked.returncode == 0 and "not found" not in linked.stdout,
            "Native dependencies missing",
        )
        require(
            "libpython" not in linked.stdout and "libtorch_python" not in linked.stdout,
            "Native executable links Python",
        )
        attention = re.search(r"libxinfer_attention_ops\.so => (\S+)", linked.stdout)
        require(
            attention and Path(attention[1]).is_file(), "Linked native attention library missing"
        )
        report["attention_library_sha256"] = sha256(attention[1])
        valid_cases(
            args, fixtures(protocol, args.reference), env, report, pair["decode"]["vocab_size"]
        )
        rejections(args, env, pair, report)
        require(not torch.cuda.is_initialized(), "Controller initialized CUDA")
        report.update(
            passed=True,
            cli_passed=True,
            cli_bitwise=True,
            no_python_library=True,
            controller_cuda_initialized=False,
        )
    except Exception as error:
        report["error"] = dict(type=type(error).__name__, message=str(error))
        traceback.print_exc()
    write_json(args.output / "summary.json", report)
    print(f"CLI pass: {report['cli_passed']}; report: {args.output / 'summary.json'}", flush=True)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
