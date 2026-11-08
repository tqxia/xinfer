"""Export and verify the explicit-input MegaQwen custom-op prototype.

This correctness probe produces a bare AOTI package, not a shipping model bundle.
Its numerical reference is the Python custom-op path. Existing
HF acceptance failures remain separate; no timing or performance claim is made.
Caller storage checks, generated-copy inspection and warmed allocation bounds
are independent checks; generated AOTI retains guarded alignment copies. These
checks do not inspect tensor pointers inside the operator.
"""

import argparse
import hashlib
import json
import re
import zipfile
from pathlib import Path

import torch

from compiler.common.artifacts import read_json, require, sha256, write_json
from compiler.common.attention import cache_extent
from compiler.common.metadata import read_package_metadata
from compiler.common.package import compile_package
from compiler.common.tests.tensors import bits_equal
from compiler.common.weights import SharedWeights, validate_shared_weights
from compiler.qwen3.megaqwen import (
    WEIGHT_SPECS,
    allocate_workspace,
    load_megaqwen_ops,
    megaqwen_c_shims,
    megaqwen_library,
    rope_tables,
)
from compiler.qwen3.packages import validate_model_metadata
from tools.common import prompt_chunks, run
from tools.validation.common.native import native_environment


def canonical_weights(pool):
    """Borrow semantic tensor views directly from an existing SharedWeights pool."""
    result = []
    for name, shape in WEIGHT_SPECS:
        view = pool.manifest["sources"].get(name)
        require(view in pool.tensors, f"Shared weights source is missing: {name}")
        tensor = pool.tensors[view]
        require(
            tuple(tensor.shape) == shape and tensor.dtype == torch.bfloat16,
            f"Unsupported MegaQwen weight shape/dtype: {name}",
        )
        require(
            tensor.device.type == "cuda" and tensor.is_contiguous(),
            f"Unsupported MegaQwen weight device/stride: {name}",
        )
        require(tensor.data_ptr() % 16 == 0, f"Unaligned MegaQwen weight: {name}")
        result.append(tensor)
    return result


class MegaQwenStep(torch.nn.Module):
    """No parameters, buffers or serialized pointer context: all tensors are inputs."""

    def forward(self, ids, weights, kv, cos, sin, workspace, extent):
        return torch.ops.xinfer.megaqwen_decode(ids, weights, kv, cos, sin, workspace, extent)


def cases_for_capacity(capacity, maximum=512):
    """Matched teacher inputs cover split thresholds, chunking, reuse and tails."""
    require(type(capacity) is int and 512 <= capacity <= 40960, "Invalid prototype capacity")
    seed = [42, 13, 151645, 0, 151643, 151935, 17]

    def tokens(length, start=0):
        return [seed[(start + index) % len(seed)] for index in range(length)]

    result = []
    for name, lengths in (
        ("initial_singleton", [1, 1, 1]),
        ("split_128", [127, 1, 1, 2, 1]),
        ("split_256", [255, 1, 1, 2, 1]),
        ("prompt_513", [512, 1, 1, 2, 1]),
        ("prompt_1025", [512, 512, 1, 1]),
    ):
        if sum(lengths) > capacity or max(lengths) > maximum:
            continue
        start, steps = 0, []
        for length in lengths:
            steps.append(tokens(length, start))
            start += length
        case = {"name": name, "steps": steps, "dump_cache": True}
        if name == "split_256":
            case["nondefault_stream"] = True
        result.append(case)
        if name == "split_128":
            result.append(dict(case, name=name + "_poison", poison_kv=3, same_as=name))
    result.append(
        {
            "name": "context_end",
            "steps": prompt_chunks(tokens(capacity - 2), maximum) + [[13], [17]],
            "check_overflow": True,
            "dump_cache": False,
        }
    )
    return result


def tensor_digest(tensor):
    data = tensor.detach().cpu().contiguous().view(torch.uint8).numpy()
    return hashlib.sha256(memoryview(data)).hexdigest()


def cache_file_digests(path, position):
    width = 8 * position * 128 * 2
    require(path.stat().st_size == 56 * width, "Incorrect native valid-prefix cache file size")
    with path.open("rb") as file:
        return [hashlib.sha256(file.read(width)).hexdigest() for _ in range(56)]


def caller_storage(inputs):
    """Record caller storage; internal temporary copies remain outside this check."""
    ids, weights, kv, cos, sin, workspace, extent = inputs
    return [tensor.data_ptr() for tensor in (ids, *weights, *kv, cos, sin, *workspace, extent)]


def check_caller_storage(inputs, previous, output):
    require(caller_storage(inputs) == previous, "Execution replaced caller-owned storage")
    require(output.data_ptr() not in previous, "Logits alias a caller input")


def inspect_generated_copies(sources):
    """Conservatively classify generated calls, retaining alignment fallbacks.

    Only the compiler's simple 16-byte input-alignment repair is recognized.
    Other clone/copy calls remain unexpected, including unfamiliar guarded forms.
    This is a syntactic check of generated code, not general C++ flow analysis.
    """
    comments_and_strings = re.compile(
        r'//[^\n]*|/\*[\s\S]*?\*/|"(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])*\''
    )
    call = re.compile(
        r"(?P<name>[A-Za-z_]\w*(?:(?:::|\.|->)[A-Za-z_]\w*)*)\s*(?:<[^;(){}]*>)?\s*\("
    )
    input_assignment = re.compile(
        r"auto\s+(\w+)\s*=\s*std::move\s*\(\s*inputs\s*\[\s*(\d+)\s*\]\s*\)\s*;"
    )
    alignment_guard = re.compile(
        r"if\s*\(\s*\(\s*reinterpret_cast\s*<\s*(?:std::)?uintptr_t\s*>\s*"
        r"\(\s*(?P<input>\w+)\.data_ptr\s*\(\s*\)\s*\)\s*&\s*"
        r"(?:\(\s*16\s*-\s*1\s*\)|\(?\s*15\s*\)?)\s*\)\s*!=\s*0\s*\)\s*"
        r"\{(?P<body>[^{}]*)\}"
    )
    guarded, unexpected = [], []
    for name, source in sources.items():
        # Preserve offsets and line numbers while excluding comments/literals.
        code = comments_and_strings.sub(lambda match: re.sub(r"[^\n]", " ", match.group()), source)
        inputs = {match.group(1): int(match.group(2)) for match in input_assignment.finditer(code)}
        recognized = {}
        for guard in alignment_guard.finditer(code):
            tensor = guard.group("input")
            if tensor not in inputs:
                continue
            body = re.fullmatch(
                r"\s*(?:AOTI_TORCH_WARN\s*\([^;]*\)\s*;\s*)?"
                r"AtenTensorHandle\s+(?P<aligned>\w+)\s*;\s*"
                r"aoti_torch_clone_preserve_strides\s*\(\s*"
                + re.escape(tensor)
                + r"\s*,\s*&\s*(?P=aligned)\s*\)\s*;\s*"
                + re.escape(tensor)
                + r"\s*=\s*std::move\s*\(\s*RAIIAtenTensorHandle\s*\(\s*"
                r"(?P=aligned)\s*\)\s*\)\s*;\s*",
                guard.group("body"),
            )
            if body is not None:
                offset = guard.start("body") + guard.group("body").index(
                    "aoti_torch_clone_preserve_strides"
                )
                recognized[offset] = {
                    "input": tensor,
                    "input_index": inputs[tensor],
                    "alignment_bytes": 16,
                    "guard_line": source.count("\n", 0, guard.start()) + 1,
                }
        for match in call.finditer(code):
            callee = match.group("name")
            if not any(token in callee.lower() for token in ("clone", "copy", "memcpy")):
                continue
            line = source.count("\n", 0, match.start()) + 1
            finding = {
                "file": name,
                "line": line,
                "callee": callee,
                "source": source.splitlines()[line - 1].strip(),
            }
            if match.start() in recognized:
                guarded.append({**finding, **recognized[match.start()]})
            else:
                unexpected.append(finding)
    return {
        "guarded_alignment_fallback_copies": {"count": len(guarded), "calls": guarded},
        "unexpected_clone_copy_calls": unexpected,
        "zero_copy_scope": "generated alignment guards only; execution of their fallback is not observed",
    }


def require_no_unexpected_copies(report):
    require(
        not report["unexpected_clone_copy_calls"],
        "Generated AOTI contains unclassified or unconditional clone/copy calls: "
        + json.dumps(report["unexpected_clone_copy_calls"]),
    )


def inspect_megaqwen_dispatch(sources):
    """Verify the decode wrapper calls the packaged C shim directly."""
    pattern = re.compile(
        r"AOTI_TORCH_ERROR_CODE_CHECK\s*\(\s*"
        r"(?P<callee>aoti_torch_cuda_megaqwen_decode|aoti_torch_proxy_executor_call_function)\s*\("
    )
    calls, proxies = [], []
    for name, source in sources.items():
        for match in pattern.finditer(source):
            row = {"file": name, "line": source.count("\n", 0, match.start()) + 1}
            (calls if match["callee"] == "aoti_torch_cuda_megaqwen_decode" else proxies).append(row)
    require(
        len(calls) == 1 and not proxies, "Expected one direct MegaQwen C-shim call and no proxy"
    )
    return {"megaqwen_c_shim_calls": calls, "proxy_executor_calls": proxies}


def export_package(inputs, output):
    module = MegaQwenStep().eval()
    capacity = inputs[2][0].shape[2]
    extent = torch.export.Dim("megaqwen_extent", min=2, max=capacity + 1)
    shapes = (
        {},
        [{} for _ in inputs[1]],
        [{} for _ in inputs[2]],
        {},
        {},
        [{} for _ in inputs[5]],
        {0: extent},
    )
    exported = torch.export.export(module, inputs, dynamic_shapes=shapes, strict=False)
    exported.validate()
    require(
        not exported.state_dict and not exported.constants,
        "The prototype must not serialize weights, scratch or pointer context",
    )
    (output / "exported.graph.txt").write_text(exported.graph_module.code)
    (output / "exported.signature.txt").write_text(str(exported.graph_signature))
    targets = [node.target for node in exported.graph.nodes if node.op == "call_function"]
    require(
        targets.count(torch.ops.xinfer.megaqwen_decode.default) == 1,
        "Expected one opaque MegaQwen operator",
    )
    package = output / "megaqwen_decode.pt2"
    compile_package(
        exported,
        package,
        inductor_configs={},
        native_libraries=(megaqwen_library(),),
        custom_op_shims=megaqwen_c_shims(),
    )
    compiled = torch._inductor.aoti_load_package(str(package))
    require(not compiled.get_constant_fqns(), "The prototype contains serialized constants")
    sources = {}
    with zipfile.ZipFile(package) as archive:
        for name in archive.namelist():
            if name.endswith((".cpp", ".h")):
                text = archive.read(name).decode()
                destination = output / "generated" / name
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_text(text)
                sources[name] = text
    require(bool(sources), "AOTI package contains no generated C++ source to inspect")
    report = {
        "package": str(package),
        "sha256": sha256(package),
        "constants": [],
        "runtime_tensor_inputs": 383,
        "source_files": list(sources),
        **inspect_generated_copies(sources),
        **inspect_megaqwen_dispatch(sources),
    }
    write_json(output / "export.json", report)
    require_no_unexpected_copies(report)
    return compiled, report


def check_position_zero(compiled, weights, kv, cos, sin, workspace):
    """Check direct and compiled singleton calls at the export lower bound."""
    ids = torch.tensor([[13]], dtype=torch.int64, device=kv[0].device)
    inputs = (ids, weights, kv, cos, sin, workspace, cache_extent(1))

    def reset():
        for tensor in kv:
            tensor.zero_()
        for tensor in workspace:
            tensor.fill_(float("nan"))

    expected, prefixes = None, None
    addresses = caller_storage(inputs)
    for function in (MegaQwenStep(), compiled):
        reset()
        extent_before = inputs[-1].clone()
        logits = function(*inputs)
        check_caller_storage(inputs, addresses, logits)
        require(bits_equal(inputs[-1], extent_before), "Position-zero extent was mutated")
        require(ids.item() == 13, "Position-zero token was mutated")
        require(bool(torch.isfinite(logits).all()), "Nonfinite direct position-zero logits")
        for tensor in kv:
            require(bool(tensor[:, :, 1:].eq(0).all()), "Position-zero call changed future KV")
        current = logits.cpu().clone()
        hashes = [tensor_digest(tensor[:, :, :1]) for tensor in kv]
        if expected is None:
            expected, prefixes = current, hashes
        else:
            require(
                bits_equal(current, expected) and hashes == prefixes,
                "Direct op/AOTI position-zero results differ bitwise",
            )
        del logits

    # The preceding AOTI invocation warms its lazy runtime state. Measure memory
    # allocation only, with no events or timing, to catch transient cache clones
    # independently of generated-source inspection and caller storage checks.
    reset()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    allocated = torch.cuda.memory_allocated()
    logits = compiled(*inputs)
    torch.cuda.synchronize()
    peak_delta = torch.cuda.max_memory_allocated() - allocated
    check_caller_storage(inputs, addresses, logits)
    require(bits_equal(logits.cpu(), expected), "Warm AOTI replay changed position-zero logits")
    cache_bytes = sum(tensor.numel() * tensor.element_size() for tensor in kv)
    limit = max(8 * 1024 * 1024, cache_bytes // 8)
    require(peak_delta < limit, "Warm AOTI allocation suggests a full-cache clone")
    return {
        "passed": True,
        "position": 0,
        "extent_elements": 2,
        "caller_owned_storage_preserved": True,
        "fresh_output_storage": True,
        "warm_aoti_peak_allocated_delta_bytes": peak_delta,
        "allocation_limit_bytes": limit,
        "cache_bytes": cache_bytes,
        "benchmark": False,
    }


def run_python_path(
    label, function, pool, weights, kv, cos, sin, workspace, cases, output, maximum
):
    """Replay on one canonical pool; only the selected singleton implementation changes."""
    prefill = pool.load("prefill")
    cache_addresses = [tensor.data_ptr() for tensor in kv]
    results = []
    for case in cases:
        fill, position, selected, retained = case.get("poison_kv", 0), 0, 0, []
        for tensor in kv:
            tensor.fill_(fill)
        path = output / f"{case['name']}.{label}.f32"
        rows = []
        with path.open("wb") as file:
            for tokens in case["steps"]:
                require(
                    1 <= len(tokens) <= maximum and position + len(tokens) <= kv[0].shape[2],
                    "Invalid prototype teacher step",
                )
                ids = torch.tensor([tokens], dtype=torch.int64, device=kv[0].device)
                end = position + len(tokens)
                extent = cache_extent(end)
                extent.fill_(255)
                if len(tokens) == 1:
                    before = [tensor.clone() for tensor in kv]
                    for tensor in workspace:
                        tensor.fill_(float("nan") if label == "python_op" else -9876.5)
                    inputs = (ids, weights, kv, cos, sin, workspace, extent)
                    addresses = caller_storage(inputs)
                    extent_before = extent.clone()
                    logits = function(*inputs)
                    check_caller_storage(inputs, addresses, logits)
                    require(bits_equal(extent, extent_before), "Extent input was mutated")
                    for actual, original in zip(kv, before):
                        require(
                            bits_equal(actual[:, :, :position], original[:, :, :position]),
                            "Custom operator overwrote past KV",
                        )
                        require(
                            bits_equal(actual[:, :, end:], original[:, :, end:]),
                            "Custom operator overwrote future KV",
                        )
                        require(
                            not bits_equal(
                                actual[:, :, position:end], original[:, :, position:end]
                            ),
                            "Custom operator omitted a layer's KV append",
                        )
                    del before
                    selected += 1
                else:
                    positions = torch.arange(position, end, device=ids.device)
                    logits = prefill(ids, positions, tuple(kv), extent)
                require(
                    logits.shape == (1, 151936) and logits.dtype == torch.bfloat16,
                    "Incorrect prototype logits shape/dtype",
                )
                require(bool(torch.isfinite(logits).all()), "Nonfinite prototype logits")
                require(torch.equal(ids.cpu(), torch.tensor([tokens])), "Token IDs were mutated")
                require(
                    [tensor.data_ptr() for tensor in kv] == cache_addresses,
                    "Prototype replaced caller-owned KV",
                )
                for tensor in kv:
                    require(bool(torch.isfinite(tensor).all()), "Nonfinite prototype KV")
                    require(bool(tensor[:, :, end:].eq(fill).all()), "Future KV poison changed")
                for previous, expected in retained:
                    require(
                        previous.data_ptr() != logits.data_ptr()
                        and bits_equal(previous.cpu(), expected),
                        "A later call reused or changed old logits",
                    )
                cpu = logits.cpu().clone()
                retained.append((logits, cpu))
                rows.append(cpu)
                file.write(cpu.float().numpy().tobytes())
                position = end
        hashes = [tensor_digest(tensor[:, :, :position]) for tensor in kv]
        if case.get("check_overflow"):
            require(position == kv[0].shape[2], "Overflow fixture did not fill the context")
            rejected = False
            try:
                function(
                    torch.tensor([[13]], dtype=torch.int64, device=kv[0].device),
                    weights,
                    kv,
                    cos,
                    sin,
                    workspace,
                    cache_extent(position + 1),
                )
            except RuntimeError:
                rejected = True
            require(rejected, "Custom operator accepted an extent beyond KV capacity")
            require(
                hashes == [tensor_digest(tensor) for tensor in kv],
                "Rejected extent changed KV state",
            )
        results.append(
            {
                "name": case["name"],
                "position": position,
                "rows": len(rows),
                "selected_decode_calls": selected,
                "logits_file": str(path),
                "valid_kv_sha256": hashes,
                "caller_owned_storage_preserved": True,
                "fresh_output_storage": True,
            }
        )
    by_name = {row["name"]: row for row in results}
    for case in cases:
        if "same_as" in case:
            compare_paths(
                [by_name[case["same_as"]]], [dict(by_name[case["name"]], name=case["same_as"])]
            )
    return results


def compare_paths(left, right):
    require(
        [row["name"] for row in left] == [row["name"] for row in right],
        "Prototype paths cover different cases",
    )
    for a, b in zip(left, right):
        require(
            a["position"] == b["position"] and a["rows"] == b["rows"],
            "Prototype path positions or row counts differ",
        )
        require(
            sha256(a["logits_file"]) == sha256(b["logits_file"]),
            f"Prototype logits differ bitwise: {a['name']}",
        )
        require(
            a["valid_kv_sha256"] == b["valid_kv_sha256"],
            f"Prototype KV differs bitwise: {a['name']}",
        )


def verify(args, report):
    pair = read_package_metadata(args.packages, model_type="qwen3")
    for metadata in pair.values():
        validate_model_metadata(metadata)
    validate_shared_weights(args.packages, pair["prefill"], pair["decode"])
    load_megaqwen_ops(args.op_library)
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    pool = SharedWeights(args.packages, pair["decode"]["shared_weights"])
    weights = canonical_weights(pool)
    capacity = pair["decode"]["cache_length"]
    maximum = pair["prefill"]["token_range"][1]
    kv = [
        torch.zeros(spec["shape"], dtype=torch.bfloat16, device="cuda")
        for spec in pair["decode"]["state"]
    ]
    workspace = allocate_workspace()
    cos, sin = rope_tables(capacity)
    ids = torch.tensor([[13]], dtype=torch.int64, device="cuda")
    inputs = (ids, weights, kv, cos, sin, workspace, cache_extent(33))
    compiled, export = export_package(inputs, args.output)
    report.update(
        capacity=capacity,
        export=export,
        caller_weight_pointers=[t.data_ptr() for t in weights],
        caller_kv_pointers=[t.data_ptr() for t in kv],
        caller_workspace_pointers=[t.data_ptr() for t in workspace],
    )
    if args.export_only:
        report["export_only"] = True
        return
    report["direct_position_zero"] = check_position_zero(compiled, weights, kv, cos, sin, workspace)
    if not args.skip_opcheck:
        report["opcheck"] = torch.library.opcheck(torch.ops.xinfer.megaqwen_decode.default, inputs)
    cases = cases_for_capacity(capacity, maximum)
    write_json(args.output / "cases.json", {"cases": cases})
    rope_before = [tensor_digest(cos), tensor_digest(sin)]
    paths = {}
    for label, function in (("python_op", MegaQwenStep()), ("python_aoti", compiled)):
        paths[label] = run_python_path(
            label, function, pool, weights, kv, cos, sin, workspace, cases, args.output, maximum
        )
    compare_paths(paths["python_op"], paths["python_aoti"])
    require(rope_before == [tensor_digest(cos), tensor_digest(sin)], "RoPE tables were mutated")
    report["python_paths"] = paths
    require_no_unexpected_copies(export)
    # The native process reloads the same verified archive and does not load Python.
    # Drop Python GPU owners first so the probe does not duplicate a full model/cache.
    del compiled, inputs, kv, workspace, weights, pool
    torch.cuda.empty_cache()
    destination = args.output / "native"
    run(
        [
            args.native.resolve(),
            args.packages.resolve(),
            Path(export["package"]).resolve(),
            (args.output / "cases.json").resolve(),
            destination.resolve(),
        ],
        native_environment(),
        args.output / "native.log",
    )
    native = read_json(destination / "summary.json")
    require(native.get("passed") is True, "Native custom-op/AOTI contract failed")
    require(
        native.get("caller_owned_storage_preserved") is True,
        "Native caller-owned storage check is missing",
    )
    for case, python in zip(cases, paths["python_op"]):
        for backend in ("op", "aoti"):
            require(
                sha256(destination / f"{case['name']}.{backend}.f32")
                == sha256(python["logits_file"]),
                f"Native/Python {backend} logits differ: {case['name']}",
            )
            if case.get("dump_cache"):
                require(
                    cache_file_digests(
                        destination / f"{case['name']}.{backend}.kv.bf16", python["position"]
                    )
                    == python["valid_kv_sha256"],
                    f"Native/Python {backend} KV differs: {case['name']}",
                )
    report["native"] = native
    report["native_python_op_logits_bitwise"] = True
    report["native_python_selected_kv_bitwise"] = True
    report["passed"] = True


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packages", type=Path, required=True)
    parser.add_argument("--op-library", type=Path)
    parser.add_argument("--native", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--export-only", action="store_true")
    parser.add_argument("--skip-opcheck", action="store_true")
    args = parser.parse_args(argv)
    require(args.export_only or args.native is not None, "Provide the native correctness probe")
    require(not args.output.exists(), "Choose an absent output directory")
    args.output.mkdir(parents=True)
    report = {
        "passed": False,
        "benchmark": False,
        "numerical_acceptance": False,
        "reference": "python_custom_op",
        "packages": str(args.packages.resolve()),
    }
    try:
        with torch.inference_mode():
            verify(args, report)
    except Exception as error:
        report["error"] = str(error)
        raise
    finally:
        write_json(args.output / "summary.json", report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
