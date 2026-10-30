"""Validate shape-based GDN dispatch through strict export, AOTI and C++."""

import argparse
import hashlib
import importlib.metadata
import inspect
import os
from pathlib import Path
import re
import subprocess
import traceback
import zipfile

import torch
from transformers.models.qwen3_5 import modeling_qwen3_5

from compiler.common.artifacts import ROOT, read_json, sha256, write_json
from compiler.common.validation import bits_equal, compare_file, cpu_copy
from compiler.qwen35.config import ARTIFACT_DIR, configure
from compiler.qwen35.diagnostics.block import MODEL_DIR
from compiler.qwen35.kernels.recurrent import SHORT_THRESHOLD, fused as dispatch, long, short
from compiler.qwen35.kernels.chunked import CHUNK_SIZE, TRANSFORM_TILE
from compiler.qwen35.tests.validate_fused import (
    ABS_LIMITS, LIMITS, PARTITIONS, compare, make_cases, reference, run_cases, write_cases,
)

STATE_FIELDS = ("conv", "state", "weight", "A_log", "dt_bias")
SHORT_OP = "xinfer.delta_cond_short.default"
LONG_OP = "xinfer.delta_cond_long.default"
LONG_KERNELS = {"prepare_kernel", "_solve", "_transforms", "_states", "_outputs"}


class Conditional(torch.nn.Module):
    def forward(self, qkv, b, a, conv, state, weight, A_log, dt_bias):
        return dispatch(qkv, b, a, conv, state, weight, A_log, dt_bias)


def dispatch_cases(cases):
    """Extend frozen original fixtures by slicing; consume no additional RNG."""
    extra = []
    for dtype in ("bf16", "fp32"):
        for state in ("zero", "state"):
            source = next(c for c in cases if c["name"] == f"{dtype}-t512-{state}")
            for tokens in (SHORT_THRESHOLD - 1, SHORT_THRESHOLD, SHORT_THRESHOLD + 1):
                extra.append(dict(source, name=f"{dtype}-cond-t{tokens}-{state}",
                                  steps=[tuple(x[:, :tokens].clone() for x in source["steps"][0])]))
        source = next(c for c in cases if c["name"] == f"{dtype}-whole")
        lengths = (SHORT_THRESHOLD - 1, SHORT_THRESHOLD + 1,
                   SHORT_THRESHOLD, SHORT_THRESHOLD + 2, 1)
        lengths += (512 - sum(lengths),)
        assert min(lengths) > 0 and sum(lengths) == 512
        data = [x.split(lengths, dim=1) for x in source["steps"][0]]
        alternate = dict(source, name=f"{dtype}-cond-alternate", steps=list(zip(*data)))
        extra.extend((alternate, dict(alternate, name=f"{dtype}-cond-alternate-repeat")))
        # Original carry already covers long prefill followed by 64 decode calls.
        carry = next(c for c in cases if c["name"] == f"{dtype}-carry")
        extra.append(dict(carry, name=f"{dtype}-cond-short-carry", steps=[
            tuple(x[:, :SHORT_THRESHOLD - 1].clone() for x in carry["steps"][0]),
            *carry["steps"][1:],
        ]))
    return cases + extra


def inputs_for(case):
    return tuple(x.cuda().contiguous() for x in
                 (*case["steps"][0], *(case[key] for key in STATE_FIELDS)))


def selected(*inputs):
    """Explicit branch control, independent of torch.cond lowering."""
    branch = short if inputs[0].shape[1] <= SHORT_THRESHOLD else long
    return branch(*(x.contiguous() for x in inputs))


def graph_evidence(exported, mode, phase, directory):
    """Check symbolic control flow and retain each branch's graph/kernel list."""
    from torch._higher_order_ops.triton_kernel_wrap import kernel_side_table

    exported.validate()
    assert not exported.graph_signature.user_inputs_to_mutate
    path = directory / f"{mode}-{phase}.graph.txt"
    path.write_text(exported.graph_module.print_readable(print_output=False) + "\n")
    targets, kernels, conditions = set(), set(), []
    for module_name, module in exported.graph_module.named_modules():
        if not isinstance(module, torch.fx.GraphModule):
            continue
        for node in module.graph.nodes:
            targets.add(str(node.target))
            if node.target == torch.ops.higher_order.cond:
                predicate = node.args[0]
                assert isinstance(predicate, torch.fx.Node), "Constant cond predicate"
                value = predicate.meta.get("val")
                assert isinstance(value, torch.SymBool), f"Non-symbolic predicate: {value}"
                assert str(value) not in ("True", "False"), "Specialized cond predicate"
                conditions.append(dict(module=module_name, predicate=str(predicate), value=str(value),
                                       branches=[str(x.target) for x in node.args[1:3]]))
            if "triton_kernel_wrapper" in str(node.target):
                kernel = kernel_side_table.get_kernel(node.kwargs["kernel_idx"])
                while not hasattr(kernel, "__name__"):
                    kernel = kernel.fn
                kernels.add(kernel.__name__)
    if mode.endswith("prefill"):
        assert len(conditions) == 1, f"Expected one dynamic condition, got {conditions}"
        if phase == "export":
            assert {SHORT_OP, LONG_OP} <= targets, "A branch operator disappeared"
        else:
            assert {"fused_kernel", *LONG_KERNELS} <= kernels, "A decomposed branch disappeared"
    else:
        assert not conditions, "Decode should directly use the fused kernel"
        if phase == "export":
            assert SHORT_OP in targets and LONG_OP not in targets
        else:
            assert kernels == {"fused_kernel"}
    return dict(graph=str(path), conditions=conditions, kernels=sorted(kernels), targets=sorted(targets))


def package_evidence(path, mode, directory):
    """Audit actual packaged cubins and generated native control flow."""
    wrappers, cubins = [], []
    with zipfile.ZipFile(path) as package:
        for member in package.namelist():
            if member.endswith(".cubin"):
                data = package.read(member)
                cubins.append(dict(member=member, bytes=len(data), sha256=hashlib.sha256(data).hexdigest()))
            elif member.endswith(".wrapper.cpp"):
                text = package.read(member).decode()
                destination = directory / f"{mode}-{len(wrappers)}.wrapper.cpp"
                destination.write_text(text)
                wrappers.append((destination, text))
    assert wrappers, f"No packaged C++ wrapper in {path}"
    text = "\n".join(source for _, source in wrappers)
    expected = {"fused_kernel", *LONG_KERNELS} if mode.endswith("prefill") else {"fused_kernel"}
    kernel_names = {name for name in {"fused_kernel", *LONG_KERNELS}
                    if re.search(rf"\bcall_{re.escape(name)}_\d+\s*\(", text)}
    assert len(cubins) == len(expected), f"Unexpected cubin count in {mode}: {len(cubins)}"
    assert kernel_names == expected, f"Unexpected native kernels: {kernel_names}"
    predicates = [line.strip() for line in text.splitlines()
                  if re.search(rf"\bif\s*\([^;\n]*<=\s*{SHORT_THRESHOLD}(?:L|\b)", line)]
    if mode.endswith("prefill"):
        assert predicates, "Missing native shape predicate"
    else:
        assert not predicates
    return dict(cubins=cubins, kernels=sorted(kernel_names),
                wrappers=[str(p) for p, _ in wrappers], native_predicates=predicates)


def compile_packages(cases, output, report):
    packages = {}
    destination, evidence = output / "packages", output / "graphs"
    destination.mkdir()
    evidence.mkdir()
    for dtype in ("bf16", "fp32"):
        for mode, tokens in (("decode", 1), ("prefill", 32)):
            name = f"{dtype}-{mode}"
            case = next(c for c in cases if c["name"] == f"{dtype}-t{tokens}-state")
            sample = inputs_for(case)
            if mode == "prefill":
                for op in (torch.ops.xinfer.delta_cond_short.default, torch.ops.xinfer.delta_cond_long.default):
                    report.setdefault("opcheck", {})[f"{dtype}-{op}"] = torch.library.opcheck(
                        op, sample, test_utils=("test_schema", "test_faketensor"))
            dynamic = None
            if mode == "prefill":
                length = torch.export.Dim("tokens", min=2, max=512)
                dynamic = tuple([{1: length}] * 3 + [None] * 5)
            exported = torch.export.export(Conditional(), sample, dynamic_shapes=dynamic, strict=True)
            entry = dict(dtype=dtype, mode=mode,
                         range_constraints=str(exported.range_constraints),
                         export=graph_evidence(exported, name, "export", evidence))
            report["packages"].append(entry)
            decomposed = exported.run_decompositions({}, decompose_custom_triton_ops=True)
            entry["decomposed"] = graph_evidence(decomposed, name, "decomposed", evidence)
            path = destination / f"{name}.pt2"
            torch._inductor.aoti_compile_and_package(exported, package_path=str(path),
                                                    inductor_configs={"emulate_precision_casts": True})
            entry.update(sha256=sha256(path), evidence=package_evidence(path, name, evidence))
            packages[dtype, mode] = torch._inductor.aoti_load_package(str(path))
            print(f"Compiled {name}: {len(entry['evidence']['cubins'])} cubins", flush=True)
    return packages


def check_repeats(results, cases):
    for values in results.values():
        for dtype in ("bf16", "fp32"):
            for original, repeat in (("carry", "repeat"), ("split", "split-repeat"),
                                     ("cond-alternate", "cond-alternate-repeat")):
                length = next(len(c["steps"]) for c in cases if c["name"] == f"{dtype}-{original}")
                for step in range(length):
                    assert all(bits_equal(a, b) for a, b in zip(
                        values[f"{dtype}-{original}", step], values[f"{dtype}-{repeat}", step]))


def check_streams_and_layouts(cases, aoti):
    checks = []
    for dtype in ("bf16", "fp32"):
        for tokens in (SHORT_THRESHOLD - 1, SHORT_THRESHOLD + 1):
            case = next(c for c in cases if c["name"] == f"{dtype}-cond-t{tokens}-state")
            sample = inputs_for(case)
            for engine, call in (("dispatch", dispatch), ("aoti", aoti)):
                expected = cpu_copy(call(*sample))
                stream = torch.cuda.Stream()
                stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(stream):
                    actual = call(*(x.clone() for x in sample))
                stream.synchronize()
                assert all(bits_equal(a, b) for a, b in zip(cpu_copy(actual), expected)), \
                    f"Nondefault-stream mismatch: {dtype}, T={tokens}, {engine}"
                checks.append(dict(dtype=dtype, tokens=tokens, engine=engine, nondefault_stream=True))
            # The operator normalizes strides; exported packages have a contiguous-input ABI.
            expected = cpu_copy(dispatch(*sample))
            strided = list(sample)
            backing = sample[0].new_empty((*sample[0].shape[:-1], sample[0].shape[-1] * 2))
            backing[..., ::2].copy_(sample[0])
            strided[0], strided[5] = backing[..., ::2], sample[5].t().contiguous().t()
            assert not strided[0].is_contiguous() and not strided[5].is_contiguous()
            assert all(bits_equal(a, b) for a, b in zip(cpu_copy(dispatch(*strided)), expected)), \
                f"Noncontiguous-input mismatch: {dtype}, T={tokens}"
            checks.append(dict(dtype=dtype, tokens=tokens, engine="dispatch", noncontiguous_inputs=True))
    return checks


def check_cpp(binary, output, results):
    env = {k: os.environ[k] for k in ("CUDA_VISIBLE_DEVICES", "LD_LIBRARY_PATH") if k in os.environ}
    env.update(PATH="/usr/bin:/bin", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", PYTHONHOME="/nonexistent-python-home")
    native = output / "cpp-aoti"
    child = subprocess.run([str(binary.resolve()), str(output.resolve()), str(native.resolve()),
                            str((output / "packages").resolve())], env=env, text=True,
                           capture_output=True, timeout=1200)
    (output / "cpp-aoti.log").write_text(child.stdout + child.stderr)
    child.check_returncode()
    for (name, step), tensors in results.items():
        compare_file(native / f"{name}-{step}.bin", tensors)
    return dict(**read_json(native / "native.json"), bitwise_tensors=3 * len(results))


def validate(args, report):
    configure()
    assert CHUNK_SIZE == 64, "Retain the fixed HF chunk-64 control"
    torch.manual_seed(35)
    original = make_cases(args.block_dir)
    original_names = {c["name"] for c in original}
    cases = dispatch_cases(original)
    write_cases(args.output, cases)
    report.update(original_cases=len(original), cases=len(cases), calls=sum(len(c["steps"]) for c in cases),
                  checkpoint_case="bf16-checkpoint" in original_names,
                  binary_sha256=sha256(args.binary),
                  hf_source_sha256=sha256(inspect.getfile(modeling_qwen3_5)))
    report["fixture_sha256"] = {p.name: sha256(p) for p in sorted(args.output.glob("*.bin"))}
    packages = compile_packages(cases, args.output, report)
    aoti = lambda *x: packages["bf16" if x[0].dtype == torch.bfloat16 else "fp32",
                                "decode" if x[0].shape[1] == 1 else "prefill"](*x)
    calls = dict(dispatch=dispatch, branch=selected, aoti=aoti,
                 hf=reference, hf_chunk64=lambda *x: reference(*x, chunked=True), fp32=reference,
                 b3=lambda *x: (short if x[0].shape[1] == 1 else long)(*x))
    results = {}
    for name, call in calls.items():
        print(f"Running {name}: {len(cases)} cases", flush=True)
        results[name] = run_cases(call, cases, fp32=name == "fp32")
    check_repeats(results, cases)
    report["stream_layout_checks"] = check_streams_and_layouts(cases, aoti)
    for expected in ("branch", "aoti", "hf", "hf_chunk64", "fp32", "b3"):
        compare(report, f"dispatch_vs_{expected}", results["dispatch"], results[expected],
                exact_tensors=(0, 1, 2) if expected in ("branch", "aoti") else ())
    compare(report, "hf_vs_fp32", results["hf"], results["fp32"])
    for engine, values in results.items():
        for dtype in ("bf16", "fp32"):
            parts = [values[f"{dtype}-split", i] for i in range(len(PARTITIONS))]
            key = (f"{dtype}-partition", 0)
            compare(report, f"{engine}_partition",
                    {key: (torch.cat([p[0] for p in parts], dim=1), *parts[-1][1:])},
                    {key: values[f"{dtype}-whole", 0]}, exact_tensors=(1,))
    report["cpp"] = check_cpp(args.binary, args.output, results["aoti"])
    regression = {"dispatch_vs_branch", "dispatch_vs_aoti", "dispatch_vs_hf", "dispatch_vs_hf_chunk64",
                  "dispatch_partition", "branch_partition", "aoti_partition"}
    report["regression_failures"] = [r for r in report["failures"] if r["comparison"] in regression or
                                     (r["comparison"] == "dispatch_vs_fp32" and r["case"].startswith("fp32-"))]
    report["dispatch_passed"] = not report["regression_failures"]
    report["fp32_failure_counts"] = {
        engine: sum(r["comparison"] == f"{engine}_vs_fp32" for r in report["failures"])
        for engine in ("dispatch", "hf")}
    report["original_fp32_failure_counts"] = {
        engine: sum(r["comparison"] == f"{engine}_vs_fp32" and r["case"] in original_names
                    for r in report["failures"]) for engine in ("dispatch", "hf")}
    report["fp32_acceptance"] = not any(report["fp32_failure_counts"].values())
    report["checks"] = dict(inputs_unchanged=True, outputs_do_not_alias_inputs=True,
                           reset_repeat_bitwise=True, finite_shapes_dtypes=True,
                           jit_python_aoti_cpp_bitwise=all(r["bitwise"] for r in report["comparisons"]
                                                         if r["comparison"] == "dispatch_vs_aoti"),
                           mixed_length_requests=True)
    # B3 is a diagnostic algorithm control: route-vs-B3 differences are retained,
    # not promoted to a new acceptance policy. HF/FP32 failures remain visible too.
    report["passed"] = not report["failures"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, default=Path("work/qwen35/build/qwen35_fused"))
    parser.add_argument("--block-dir", type=Path, default=MODEL_DIR)
    parser.add_argument("--output", type=Path, default=ARTIFACT_DIR / "cond-validation")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    report = dict(seed=35, torch=torch.__version__, triton=importlib.metadata.version("triton"),
                  short_threshold=SHORT_THRESHOLD, chunk_size=CHUNK_SIZE, transform_tile=TRANSFORM_TILE,
                  limits=LIMITS, absolute_limits=ABS_LIMITS, hf_chunk_reference=64,
                  partition_acceptance=dict(core="numerical", conv="bitwise", recurrent="numerical"),
                  source_sha256={name: sha256(ROOT / name) for name in (
                      "compiler/qwen35/tests/validate_cond.py", "compiler/qwen35/tests/validate_fused.py",
                      "compiler/qwen35/kernels/recurrent.py",
                      "compiler/qwen35/kernels/chunked.py", "runtime/qwen35/tests/fused.cpp")},
                  comparisons=[], failures=[], packages=[], passed=False, dispatch_passed=False)
    try:
        with torch.inference_mode():
            validate(args, report)
    except Exception as error:
        report.update(passed=False, dispatch_passed=False,
                      error=dict(type=type(error).__name__, message=str(error), traceback=traceback.format_exc()))
        traceback.print_exc()
    write_json(args.output / "summary.json", report)
    print({key: report[key] for key in ("passed", "dispatch_passed")})
    print(f"Numerical failures: {len(report['failures'])}; report: {args.output / 'summary.json'}")
    raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
