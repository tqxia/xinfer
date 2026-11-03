"""Compare retained full-model correctness and fresh-process GPU-memory reports.

Run with any Python 3.12 interpreter; this reporter does not import Torch.
"""

import argparse
from collections import Counter
import json
from pathlib import Path
from statistics import median
import zipfile

from compiler.common.artifacts import read_json, require, sha256


def differences(before, after, fields):
    def keys(rows):
        return Counter(json.dumps({key: row.get(key) for key in fields}, sort_keys=True) for row in rows)
    old, new = keys(before), keys(after)
    return dict(embedded_count=len(before), shared_count=len(after), unchanged=old == new,
                added_ids=[json.loads(item) for item in sorted((new - old).elements())],
                removed_ids=[json.loads(item) for item in sorted((old - new).elements())])


def memory_reports(directory, profile, model):
    reports = [read_json(directory / f"memory-{profile}-{index}.json") for index in (1, 2)]
    for report in reports:
        require(report.get("passed") is True and report.get("model") == model and
                report.get("measurement") == "cudaMemGetInfo", f"Invalid {model}/{profile} memory report")
        for stage, growth in (("after_loading", "loading_growth_bytes"),
                              ("after_decode", "after_decode_growth_bytes")):
            require(report[growth] == report[stage]["used_bytes"] - report["before_runtime"]["used_bytes"],
                    f"Inconsistent {model}/{profile} {growth}")
    return dict(
        samples=reports,
        median_loading_growth_bytes=median(row["loading_growth_bytes"] for row in reports),
        median_after_decode_growth_bytes=median(row["after_decode_growth_bytes"] for row in reports),
        median_after_loading_used_bytes=median(row["after_loading"]["used_bytes"] for row in reports),
        median_after_decode_used_bytes=median(row["after_decode"]["used_bytes"] for row in reports),
    )


def correctness(directory, model):
    suffix = "validation" if model == "qwen3" else "end-to-end"
    roots = {profile: directory / f"{profile}-{suffix}" for profile in ("embedded", "shared")}
    summaries = {profile: read_json(root / "summary.json") for profile, root in roots.items()}
    cases = {profile: read_json(root / "cases.json") for profile, root in roots.items()}
    digests = {profile: read_json(root / "native-output-digests.json") for profile, root in roots.items()}
    maps = {profile: digest["files"] for profile, digest in digests.items()}
    require(all(maps.values()), f"{model} output digests must not be empty")
    if model == "qwen3":
        parity = {profile: summary.get("numerical_policy", {}).get("native_python_aoti") == "bitwise" and
                  bool(summary.get("cases")) and
                  all(case["cpp_vs_aoti"]["max_abs"] == 0 for case in summary["cases"])
                  for profile, summary in summaries.items()}
        failures, greedy, tokens = "failures", "hf_greedy_differences", "hf_token_differences"
    else:
        parity = {profile: summary.get("native_bitwise") is True for profile, summary in summaries.items()}
        failures, greedy, tokens = "failures", "greedy_failures", "top1_differences"
    parity = {profile: passed and digests[profile].get("native_python_parity_verified") is True
              for profile, passed in parity.items()}
    old, new = summaries["embedded"], summaries["shared"]
    # Embedded-weight runs are retained historical evidence. Their Qwen3 reports
    # predate the common failure-array name; new validators only write `failures`.
    def failure_rows(summary):
        if model == "qwen3" and failures not in summary:
            return summary["numerical_failures"]
        return summary[failures]

    numerical = differences(failure_rows(old), failure_rows(new), ("comparison", "case", "step", "tensor", "domain"))
    greedy_differences = differences(old[greedy], new[greedy], ("comparison", "case"))
    token_differences = differences(old[tokens], new[tokens], ("comparison", "case", "step"))
    diagnostics = dict(
        embedded_validation_passed=old["passed"], shared_validation_passed=new["passed"],
        numerical_failure_cases=numerical, greedy_failure_cases=greedy_differences,
        top1_difference_cases=token_differences,
    )
    if model == "qwen3":
        diagnostics.update(embedded_hf_parity_passed=old["hf_parity_passed"],
                           shared_hf_parity_passed=new["hf_parity_passed"],
                           embedded_hf_reference_limits_passed=old["hf_reference_limits_passed"],
                           shared_hf_reference_limits_passed=new["hf_reference_limits_passed"])
    checks = dict(
        cases_identical=cases["embedded"] == cases["shared"],
        native_output_digests_identical=maps["embedded"] == maps["shared"],
        embedded_native_python_bitwise=parity["embedded"], shared_native_python_bitwise=parity["shared"],
        numerical_failure_cases_unchanged=numerical["unchanged"],
        greedy_failure_cases_unchanged=greedy_differences["unchanged"],
        top1_difference_cases_unchanged=token_differences["unchanged"],
        validation_status_unchanged=old["passed"] == new["passed"],
    )
    return dict(checks=checks, sharing_regression_passed=all(checks.values()),
                output_files=len(maps["shared"]),
                output_bytes=sum(row["bytes"] for row in maps["shared"].values()),
                numerical_status=diagnostics)


def archive_manifest(directory, model):
    metadata = {kind: read_json(directory / f"{kind}.json") for kind in ("prefill", "decode")}
    contract = metadata["decode"].get("shared_weights")
    require(contract == metadata["prefill"].get("shared_weights"),
            f"{model} shared weight contracts differ")
    require(isinstance(contract, dict) and contract.get("format") == 2 and
            contract.get("file") == "weights.pt" and isinstance(contract.get("identity"), str) and
            type(contract.get("bytes")) is int and contract["bytes"] >= 0 and
            isinstance(contract.get("sha256"), str), f"{model} shared weight metadata is invalid")
    path = directory / "weights.pt"
    require(path.is_file(), f"{model} shared weight archive is missing")
    require(path.stat().st_size == contract["bytes"], f"{model} shared weight archive checksum mismatch")
    require(sha256(path) == contract["sha256"], f"{model} shared weight archive checksum mismatch")
    with zipfile.ZipFile(path) as archive:
        records = [name for name in archive.namelist() if name.endswith("/extra/weights.json")]
        require(len(records) == 1, f"{model} shared archive has no unique embedded mapping")
        manifest = json.loads(archive.read(records[0]))
    require(isinstance(manifest, dict) and manifest.get("format") == 2,
            f"{model} shared archive has an unsupported format")
    require(manifest.get("identity_sha256") == contract["identity"],
            f"{model} shared weight identity mismatch")
    return manifest


def compare_model(root, model):
    directory = root / model
    manifest = archive_manifest(directory / "shared", model)
    cuda_bytes = sum(storage["bytes"] for storage in manifest["storages"].values() if storage["device"] == "cuda")
    require(cuda_bytes > 0, f"{model} shared archive contains no CUDA weights")
    memory = {profile: memory_reports(directory, profile, model) for profile in ("embedded", "shared")}
    loading_saved = memory["embedded"]["median_loading_growth_bytes"] - memory["shared"]["median_loading_growth_bytes"]
    decode_saved = memory["embedded"]["median_after_decode_growth_bytes"] - memory["shared"]["median_after_decode_growth_bytes"]
    memory.update(unique_cuda_weight_bytes=cuda_bytes,
                  median_loading_saved_bytes=loading_saved, median_after_decode_saved_bytes=decode_saved,
                  loading_saved_weight_set_ratio=loading_saved / cuda_bytes,
                  after_decode_saved_weight_set_ratio=decode_saved / cuda_bytes)
    memory["checks"] = dict(loading_savings_positive=loading_saved > 0,
                            loading_savings_at_least_80_percent_of_one_weight_set=loading_saved >= 0.8 * cuda_bytes,
                            after_decode_savings_positive=decode_saved > 0)
    memory["passed"] = all(memory["checks"].values())
    files = {profile: {name: (directory / profile / name).stat().st_size
                       for name in (("prefill.pt2", "decode.pt2", "weights.pt") if profile == "shared"
                                    else ("prefill.pt2", "decode.pt2"))}
             for profile in ("embedded", "shared")}
    sizes = {profile: sum(rows.values()) for profile, rows in files.items()}
    footprint = dict(files=files, total_bytes=sizes, saved_bytes=sizes["embedded"] - sizes["shared"],
                     saved_fraction=(sizes["embedded"] - sizes["shared"]) / sizes["embedded"])
    validation = correctness(directory, model)
    return dict(correctness=validation, gpu_memory=memory, artifact_footprint=footprint,
                bundle_checksum_verified=True,
                requested_scope_passed=validation["sharing_regression_passed"] and memory["passed"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = dict(artifacts=str(args.artifacts.resolve()), models={}, requested_scope_passed=False,
                  scope="Shared-weight correctness regression and GPU memory savings. Existing full-model numerical failures remain failures.")
    for model in ("qwen3", "qwen35"):
        try:
            report["models"][model] = compare_model(args.artifacts, model)
        except (OSError, ValueError, KeyError, TypeError, ZeroDivisionError) as error:
            report["models"][model] = dict(requested_scope_passed=False, error=f"{type(error).__name__}: {error}")
    report["requested_scope_passed"] = all(row["requested_scope_passed"] for row in report["models"].values())
    report["full_model_numerical_passed"] = all(
        row.get("correctness", {}).get("numerical_status", {}).get("shared_validation_passed", False)
        for row in report["models"].values())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    raise SystemExit(0 if report["requested_scope_passed"] else 1)


if __name__ == "__main__":
    main()
