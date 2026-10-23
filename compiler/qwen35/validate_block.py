"""Compare HF, the state adapter, AOTI and a Python-free C++ block runner."""

import argparse
import json
import os
import subprocess
from pathlib import Path

import torch
from transformers import DynamicCache

from block import BlockStep, HIDDEN, MODEL_DIR, PREFILL, configure, load_block, sha256, zero_state


LIMIT = 0.05
NAMES = ("output", "convolution", "recurrent")


def metrics(actual, expected, index):
    assert actual.shape == expected.shape
    assert torch.isfinite(actual).all() and torch.isfinite(expected).all()
    # Per token for output, per channel for conv, per head for recurrent state.
    width = expected.shape[-1] if index < 2 else expected.shape[-2] * expected.shape[-1]
    a, b = (t.float().reshape(-1, width) for t in (actual, expected))
    delta = a - b
    return dict(max_abs=delta.abs().max().item(),
                relative_l2=(delta.norm(dim=-1) / b.norm(dim=-1).clamp_min(1e-12)).max().item(),
                peak_relative=(delta.abs().amax(-1) / b.abs().amax(-1).clamp_min(1e-12)).max().item())


def bits_equal(a, b):
    return a.dtype == b.dtype and torch.equal(a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8))


def fixtures(directory):
    generator = torch.Generator().manual_seed(35)
    inputs = {"prefill": torch.randn(1, PREFILL, HIDDEN, generator=generator).bfloat16()}
    inputs.update({f"decode_{i}": torch.randn(1, 1, HIDDEN, generator=generator).bfloat16() for i in range(8)})
    for name, value in inputs.items():
        (directory / f"{name}.bin").write_bytes(value.view(torch.uint8).numpy().tobytes())

    def step(name):
        return dict(kind="prefill" if name == "prefill" else "decode", file=f"{name}.bin")

    chain = [step("prefill")] + [step(f"decode_{i}") for i in range(8)]
    cases = [dict(name="prefill_decode", steps=chain),
             dict(name="decode_zero", steps=[step(f"decode_{i}") for i in range(4)]),
             dict(name="prefill_continue", steps=[step("prefill"), step("prefill"), step("decode_0")]),
             dict(name="repeat", steps=chain)]
    (directory / "cases.json").write_text(json.dumps(dict(cases=cases), indent=2) + "\n")
    return cases, inputs


def hf_trace(block, config, cases, inputs, dtype):
    trace = {}
    for case in cases:
        cache = DynamicCache(config=config)
        for i, step in enumerate(case["steps"]):
            x = inputs[Path(step["file"]).stem].to(device="cuda", dtype=dtype)
            output = block(x, cache_params=cache)
            layer = cache.layers[0]
            trace[(case["name"], i)] = tuple(t.cpu().clone() for t in
                (output, layer.conv_states[0], layer.recurrent_states[0]))
    return trace


def explicit_trace(functions, cases, inputs):
    trace = {}
    for case in cases:
        conv, state = zero_state()
        for i, step in enumerate(case["steps"]):
            x = inputs[Path(step["file"]).stem].cuda()
            args = (x, conv, state)
            before = tuple(t.clone() for t in args)
            result = functions[step["kind"]](*args)
            assert len(result) == 3
            for a, b, c in zip(args, before, result):
                assert bits_equal(a, b), "Input mutation"
                assert c.shape == a.shape and c.dtype == a.dtype
            trace[(case["name"], i)] = tuple(t.cpu().clone() for t in result)
            conv, state = result[1:]
    return trace


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=MODEL_DIR)
    parser.add_argument("--packages", type=Path, default=Path("artifacts/qwen35/block"))
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("artifacts/qwen35/validation"))
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"Refusing to overwrite {args.output}")
    configure()
    args.output.mkdir(parents=True)
    cases, inputs = fixtures(args.output)
    for kind in ("prefill", "decode"):
        meta = json.loads((args.packages / f"{kind}.json").read_text())
        assert meta["torch"] == torch.__version__
        assert meta["package_sha256"] == sha256(args.packages / f"{kind}.pt2")
        assert meta["checkpoint"] == json.loads((args.model_dir / "manifest.json").read_text())
    block, config = load_block(args.model_dir)
    reference = hf_trace(block, config, cases, inputs, torch.bfloat16)
    wrapper = BlockStep(block)
    adapted = explicit_trace(dict(prefill=wrapper, decode=wrapper), cases, inputs)
    fp32_block, _ = load_block(args.model_dir, torch.float32)
    fp32 = hf_trace(fp32_block, config, cases, inputs, torch.float32)
    functions = {kind: torch._inductor.aoti_load_package(str(args.packages / f"{kind}.pt2"))
                 for kind in ("prefill", "decode")}
    compiled = explicit_trace(functions, cases, inputs)

    # The child inherits CUDA visibility/driver paths, but no Python environment.
    env = {k: os.environ[k] for k in ("CUDA_VISIBLE_DEVICES", "LD_LIBRARY_PATH") if k in os.environ}
    env.update(PATH="/usr/bin:/bin", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    command = [str(args.binary.resolve()), str(args.packages.resolve()), str(args.output.resolve()),
               str((args.output / "native").resolve())]
    child = subprocess.run(command, env=env, text=True, capture_output=True)
    (args.output / "native.log").write_text(child.stdout + child.stderr)
    child.check_returncode()
    ldd = subprocess.check_output(["ldd", str(args.binary.resolve())], text=True)
    (args.output / "ldd.txt").write_text(ldd)
    assert "libpython" not in ldd
    report = dict(limit=LIMIT, fixture="seeded hidden states; real layer-0 weights", seed=35,
                  cases=cases, comparisons=[], failures=[], binary_sha256=sha256(args.binary),
                  native=json.loads((args.output / "native/native.json").read_text()),
                  input_mutation=False, fp32_control="FP32 computation using the same checkpoint weights",
                  metric_axes=dict(output="per token", convolution="per channel", recurrent="per head"))
    for key, tensors in compiled.items():
        name, step = key
        row = dict(case=name, step=step, tensors={})
        for i, (label, value) in enumerate(zip(NAMES, tensors)):
            path = args.output / "native" / f"{name}-{step}-{i}.bin"
            native = torch.frombuffer(bytearray(path.read_bytes()), dtype=value.dtype).reshape(value.shape)
            assert bits_equal(native, value), f"C++/Python AOTI differ: {path}"
            entry = dict(cpp_aoti_bitwise=True)
            for comparison, actual, expected in (
                ("adapter_vs_hf", adapted[key][i], reference[key][i]),
                ("aoti_vs_hf", value, reference[key][i]),
                ("aoti_vs_adapter", value, adapted[key][i]),
                ("aoti_vs_fp32", value, fp32[key][i]),
                ("hf_vs_fp32", reference[key][i], fp32[key][i]),
            ):
                error = metrics(actual, expected, i)
                error["bitwise_equal"] = bits_equal(actual, expected)
                entry[comparison] = error
                if comparison in ("adapter_vs_hf", "aoti_vs_hf", "aoti_vs_adapter"):
                    if max(error["relative_l2"], error["peak_relative"]) > LIMIT:
                        report["failures"].append(dict(case=name, step=step, tensor=label,
                                                       comparison=comparison, **error))
            row["tensors"][label] = entry
        report["comparisons"].append(row)
    for i in range(len(cases[0]["steps"])):
        assert all(bits_equal(a, b) for a, b in zip(compiled[("prefill_decode", i)], compiled[("repeat", i)]))
    report["reset_repeat_bitwise"] = True
    report["passed"] = not report["failures"]
    (args.output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: report[k] for k in ("passed", "reset_repeat_bitwise", "failures")}, indent=2))
    raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
