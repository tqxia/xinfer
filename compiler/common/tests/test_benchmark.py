"""CPU checks for shared benchmark measurement, orchestration and report contracts."""

import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from compiler.common import benchmark
from compiler.common.artifacts import sha256, write_json
from compiler.common.metadata import validate_metadata_pair
from compiler.common.tests.test_metadata import metadata_pair
from compiler.qwen3 import benchmark as qwen3_benchmark
from compiler.qwen35 import benchmark as qwen35_benchmark


def sample(case, predictions=None):
    return dict(
        reset_ms=1,
        prefill_ms=2,
        first_prediction_ms=3,
        decode_ms=6,
        decode_ms_per_token=3,
        request_ms=9,
        prediction_ids=predictions or [11] * (1 + len(case["decode_tokens"])),
        consumed_positions=len(case["prompt"]) + len(case["decode_tokens"]),
    )


def worker_report(workload):
    return dict(
        cpu_threads=1,
        load_ms=10,
        initial_memory=dict(device_used_bytes=100),
        cases=[
            dict(
                name=case["name"],
                prompt_tokens=len(case["prompt"]),
                decode_steps=len(case["decode_tokens"]),
                first_request=sample(case),
                samples=[sample(case) for _ in range(workload["repeats"])],
                memory=dict(
                    peak_allocated_bytes=10,
                    peak_reserved_bytes=20,
                    sampled_peak_device_used_bytes=150,
                ),
            )
            for case in workload["cases"]
        ],
    )


class HFEagerBaseline(unittest.TestCase):
    def test_hf_owns_masks_positions_and_cache_lifecycle(self):
        import torch

        tensor = torch.tensor
        calls, returned = [], []

        def model(**kwargs):
            calls.append(kwargs)
            cache = object()
            returned.append(cache)
            return SimpleNamespace(
                logits=tensor([[[0.0, 9.0, 0.0], [0.0, 0.0, 9.0]]]),
                past_key_values=cache,
            )

        def cpu_tensor(*args, **kwargs):
            self.assertEqual(kwargs.pop("device"), "cuda")
            return tensor(*args, **kwargs)

        loader = Mock(return_value=model)
        reset, predict = benchmark.hf_eager_loader(loader)(Path("checkpoint"))
        loader.assert_called_once_with(Path("checkpoint"))
        with patch("torch.tensor", side_effect=cpu_tensor):
            self.assertEqual(predict([4, 5, 6], 0), 2)
            self.assertEqual(predict([7], 3), 2)
            self.assertEqual(predict([8, 9], 4), 2)
            reset()
            self.assertEqual(predict([10], 0), 2)
        for call, tokens in zip(calls, ([4, 5, 6], [7], [8, 9], [10])):
            self.assertEqual(
                set(call), {"input_ids", "past_key_values", "use_cache", "logits_to_keep"}
            )
            self.assertEqual(call["input_ids"].tolist(), [tokens])
            self.assertEqual(call["input_ids"].dtype, torch.int64)
            self.assertIs(call["use_cache"], True)
            self.assertEqual(call["logits_to_keep"], 1)
        self.assertIsNone(calls[0]["past_key_values"])
        self.assertIs(calls[1]["past_key_values"], returned[0])
        self.assertIs(calls[2]["past_key_values"], returned[1])
        self.assertIsNone(calls[3]["past_key_values"])

    def test_out_of_order_requests_fail_before_allocation_or_model_execution(self):
        model = Mock()
        _, predict = benchmark.hf_eager_loader(lambda _: model)(Path("checkpoint"))
        with patch("torch.tensor") as allocate:
            with self.assertRaisesRegex(ValueError, "Consumed position differs"):
                predict([4], 1)
        allocate.assert_not_called()
        model.assert_not_called()

    def test_a_missing_hf_cache_is_rejected(self):
        import torch

        model = Mock(return_value=SimpleNamespace(past_key_values=None))
        _, predict = benchmark.hf_eager_loader(lambda _: model)(Path("checkpoint"))
        with patch("torch.tensor", return_value=torch.empty(1, 1, dtype=torch.int64)):
            with self.assertRaisesRegex(ValueError, "HF model did not return a cache"):
                predict([4], 0)

    def test_both_model_hooks_use_the_shared_hf_loader(self):
        for family, module in (("qwen3", qwen3_benchmark), ("qwen35", qwen35_benchmark)):
            with self.subTest(model=family):
                model = SimpleNamespace(load_model=Mock(), configure_text=Mock())
                sentinel = object()
                with (
                    patch.dict(sys.modules, {f"compiler.{family}.model": model}),
                    patch.object(module, "hf_eager_loader", return_value=sentinel) as helper,
                    patch("compiler.qwen35.config.configure") as configure,
                ):
                    self.assertIs(module.eager_loader(), sentinel)
                helper.assert_called_once_with(model.load_model)
                model.configure_text.assert_not_called()
                if family == "qwen35":
                    configure.assert_called_once_with()
                else:
                    configure.assert_not_called()

    def test_qwen35_benchmark_preserves_sdpa_backend_selection(self):
        import torch

        def backends():
            return (
                torch.backends.cuda.flash_sdp_enabled(),
                torch.backends.cuda.mem_efficient_sdp_enabled(),
                torch.backends.cuda.math_sdp_enabled(),
                torch.backends.cuda.cudnn_sdp_enabled(),
            )

        model = SimpleNamespace(load_model=Mock(), configure_text=Mock())
        selected = backends()
        self.addCleanup(
            setattr, torch.backends.cuda.matmul, "allow_tf32", torch.backends.cuda.matmul.allow_tf32
        )
        self.addCleanup(
            setattr, torch.backends.cudnn, "allow_tf32", torch.backends.cudnn.allow_tf32
        )
        with (
            patch.dict(sys.modules, {"compiler.qwen35.model": model}),
            patch("compiler.qwen35.config.importlib.metadata.version", return_value="5.20.2"),
            patch("compiler.qwen35.config.importlib.util.find_spec", return_value=None),
            patch("torch.set_num_threads"),
        ):
            qwen35_benchmark.eager_loader()
            self.assertEqual(backends(), selected)
        model.configure_text.assert_not_called()
        model.load_model.assert_not_called()

    def test_eager_worker_identifies_the_hf_dynamic_baseline(self):
        with tempfile.TemporaryDirectory() as temporary:
            args = SimpleNamespace(
                output=Path(temporary) / "worker.json",
                workload=Path("workload"),
                model_dir=Path("checkpoint"),
            )
            load = Mock(return_value=(Mock(), Mock()))
            with (
                patch.object(benchmark, "read_json", return_value={}),
                patch.object(benchmark, "memory_stats", return_value={}),
                patch.object(benchmark, "measure_cases", return_value=[]),
                patch("torch.empty"),
                patch("torch.cuda.synchronize"),
                patch("torch.cuda.reset_peak_memory_stats"),
                patch("torch.set_num_threads"),
            ):
                benchmark.eager_worker(args, load, "eager")
            report = json.loads(args.output.read_text())
            self.assertEqual(report["eager_baseline"], "hf_dynamic_cache")
            self.assertEqual(report["profile"], "eager")


class BenchmarkContract(unittest.TestCase):
    def test_fixed_tokens_and_timing_boundaries(self):
        calls = []
        case = dict(prompt=[7, 8], decode_tokens=[0, 9])  # EOS-like 0 remains input data.

        def predict(tokens, position):
            calls.append((tokens, position))
            return 0

        with (
            patch("torch.cuda.synchronize") as sync,
            patch.object(benchmark.time, "perf_counter", side_effect=[0, 1, 3, 9]),
        ):
            result = benchmark.measure_request(lambda: calls.append("reset"), predict, case)
        self.assertEqual(calls, ["reset", ([7, 8], 0), ([0], 2), ([9], 3)])
        self.assertEqual(sync.call_count, 4)
        self.assertEqual(result["prediction_ids"], [0, 0, 0])
        self.assertEqual(result["consumed_positions"], 4)
        self.assertEqual(
            [result[key] for key in benchmark.TIMING_FIELDS], [1000, 2000, 3000, 6000, 3000, 9000]
        )

    def test_first_request_and_warmup_excluded(self):
        requests, reset_after = [], []
        case = dict(name="case", prompt=[1], decode_tokens=[2])

        def request(item):
            requests.append(item)
            return sample(item) | dict(serial=len(requests))

        workload = dict(warmup=2, repeats=2, cases=[case])
        with (
            patch.object(benchmark, "memory_stats", return_value={"device_used_bytes": 100}),
            patch(
                "torch.cuda.reset_peak_memory_stats",
                side_effect=lambda: reset_after.append(len(requests)),
            ),
        ):
            (row,) = benchmark.measure_cases(request, workload)
        self.assertEqual(len(requests), 5)
        self.assertEqual(row["first_request"]["serial"], 1)
        self.assertEqual([item["serial"] for item in row["samples"]], [4, 5])
        self.assertEqual(reset_after, [3])

    def test_warmup_predictions_and_position_are_checked(self):
        case = dict(name="case", prompt=[1], decode_tokens=[2])
        for key, value in (
            ("prediction_ids", [8, 9]),
            ("prediction_ids", [11]),
            ("consumed_positions", 1),
        ):
            bad = sample(case) | {key: value}
            with (
                self.subTest(key=key, value=value),
                patch.object(benchmark, "memory_stats", return_value={"device_used_bytes": 100}),
            ):
                replies = iter([sample(case), bad])
                with self.assertRaises(ValueError):
                    benchmark.measure_cases(
                        lambda _: next(replies), dict(warmup=1, repeats=1, cases=[case])
                    )

    def test_idle_requires_consecutive_samples_on_same_device(self):
        readings = ["0,0", "99,0"] + ["0,0"] * 5
        with (
            tempfile.TemporaryDirectory() as directory,
            patch.object(benchmark.subprocess, "check_output", side_effect=readings) as query,
            patch.object(benchmark.time, "sleep"),
        ):
            path = Path(directory) / "idle.json"
            readings = benchmark.wait_idle("GPU-test", output=path)
            self.assertEqual(len(readings), 7)
            self.assertTrue(json.loads(path.read_text())["idle"])
            self.assertTrue(all(call.args[0][2] == "GPU-test" for call in query.call_args_list))

    def test_busy_gpu_rejected_or_explicitly_provisional(self):
        with (
            patch.object(benchmark.subprocess, "check_output", return_value="99,90") as query,
            patch.object(benchmark.time, "sleep"),
        ):
            with self.assertRaisesRegex(RuntimeError, "not idle"):
                benchmark.wait_idle("GPU-test")
            self.assertEqual(query.call_count, 30)
            self.assertEqual(len(benchmark.wait_idle("GPU-test", allow_busy=True)), 1)


class SharedWeightComparison(unittest.TestCase):
    def profiles(self):
        baseline = {kind: dict(metadata=metadata) for kind, metadata in metadata_pair().items()}
        candidate = copy.deepcopy(baseline)
        for entry in candidate.values():
            entry["metadata"]["shared_weights"].update(bytes=1000128, sha256="b" * 64)
        weights = dict(
            identity_sha256="1" * 64,
            sources={
                "model.model.embed_tokens.weight": "v_" + "c" * 64,
                "model.model.layers.0.self_attn.q_proj.weight": "v_" + "d" * 64,
            },
            entries={"decode": {"constants": {"_tensor_constant0": "v_generated_a"}}},
        )
        other_weights = copy.deepcopy(weights)
        other_weights["entries"]["decode"]["constants"]["_tensor_constant0"] = "v_generated_b"
        return baseline, candidate, weights, other_weights

    def test_same_weights_with_different_archive_hashes_are_comparable(self):
        baseline, candidate, before, after = self.profiles()
        for profile in (baseline, candidate):
            validate_metadata_pair(profile["prefill"]["metadata"], profile["decode"]["metadata"])
        benchmark.validate_comparison(baseline, candidate, before, after)
        candidate["prefill"]["metadata"]["shared_weights"]["sha256"] = "e" * 64
        with self.assertRaisesRegex(ValueError, "Package metadata differs: shared_weights"):
            validate_metadata_pair(
                candidate["prefill"]["metadata"], candidate["decode"]["metadata"]
            )

    def test_changed_source_weights_are_rejected_despite_same_identity(self):
        baseline, candidate, before, after = self.profiles()
        after["sources"]["model.model.embed_tokens.weight"] = "v_" + "e" * 64
        with self.assertRaisesRegex(ValueError, "Incomparable shared weights: sources"):
            benchmark.validate_comparison(baseline, candidate, before, after)

    def test_changed_numerical_settings_are_rejected(self):
        baseline, candidate, before, after = self.profiles()
        for entry in candidate.values():
            entry["metadata"]["triton_enable_fp_fusion"] = False
        with self.assertRaisesRegex(
            ValueError, "Incomparable packages: prefill/triton_enable_fp_fusion"
        ):
            benchmark.validate_comparison(baseline, candidate, before, after)


class GpuSelection(unittest.TestCase):
    GPU = "GPU-22222222-2222-2222-2222-222222222222"
    OTHER = "GPU-33333333-3333-3333-3333-333333333333"

    def test_both_models_use_the_shared_controller(self):
        self.assertIs(qwen3_benchmark.run_benchmark, benchmark.run_benchmark)
        self.assertIs(qwen35_benchmark.run_benchmark, benchmark.run_benchmark)

    def test_explicit_index_or_uuid_overrides_inherited_visibility(self):
        for selector in ("2", self.GPU):
            with (
                self.subTest(selector=selector),
                patch.dict(os.environ, CUDA_VISIBLE_DEVICES="0"),
                patch.object(
                    benchmark.subprocess, "check_output", return_value=self.GPU + "\n"
                ) as query,
            ):
                self.assertEqual(benchmark.resolve_gpu(selector), self.GPU)
                query.assert_called_once()
                self.assertEqual(query.call_args.args[0][:3], ["nvidia-smi", "-i", selector])
                self.assertIn("--query-gpu=uuid", query.call_args.args[0])
                self.assertEqual(os.environ["CUDA_VISIBLE_DEVICES"], "0")

    def test_default_respects_cuda_visibility_and_order_without_parent_cuda(self):
        calls = []

        def query(command, **options):
            calls.append(command)
            if command[0] == sys.executable:
                self.assertEqual(command[1], "-c")
                child_env = options.get("env", os.environ)
                self.assertEqual(child_env["CUDA_VISIBLE_DEVICES"], "3,1")
                self.assertEqual(child_env["CUDA_DEVICE_ORDER"], "PCI_BUS_ID")
            else:
                self.assertEqual(command[:3], ["nvidia-smi", "-i", self.GPU])
            return self.GPU + "\n"

        with (
            patch.dict(os.environ, CUDA_VISIBLE_DEVICES="3,1", CUDA_DEVICE_ORDER="PCI_BUS_ID"),
            patch.object(benchmark.subprocess, "check_output", side_effect=query),
            patch(
                "torch.cuda.get_device_properties",
                side_effect=AssertionError("Parent must not initialize CUDA"),
            ),
        ):
            self.assertEqual(benchmark.resolve_gpu(), self.GPU)
            self.assertEqual(os.environ["CUDA_VISIBLE_DEVICES"], "3,1")
            self.assertEqual(os.environ["CUDA_DEVICE_ORDER"], "PCI_BUS_ID")
        self.assertEqual(len(calls), 2)

    def test_invalid_explicit_selections_do_not_probe_or_fall_back(self):
        for selector in ("", " ", "0,1", "MIG-example"):
            with (
                self.subTest(selector=selector),
                patch.object(benchmark.subprocess, "check_output") as query,
            ):
                with self.assertRaises(ValueError):
                    benchmark.resolve_gpu(selector)
                query.assert_not_called()

    def test_empty_multiple_or_nonphysical_uuid_results_are_rejected(self):
        for result in ("", self.GPU + "\n" + self.OTHER, "MIG-example", "not-a-gpu"):
            for selector in (None, "2"):
                with (
                    self.subTest(result=result, selector=selector),
                    patch.object(
                        benchmark.subprocess, "check_output", return_value=result
                    ) as query,
                ):
                    with self.assertRaises(ValueError):
                        benchmark.resolve_gpu(selector)
                    query.assert_called_once()

    def test_cuda_uuid_cannot_silently_resolve_to_another_physical_gpu(self):
        with patch.object(
            benchmark.subprocess, "check_output", side_effect=[self.GPU, self.OTHER]
        ) as query:
            with self.assertRaisesRegex(ValueError, "UUID does not match"):
                benchmark.resolve_gpu()
        self.assertEqual(query.call_count, 2)

    def test_missing_cuda_or_failed_smi_lookup_never_falls_back(self):
        failure = subprocess.CalledProcessError(
            1, ["device-query"], stderr="No accessible CUDA device"
        )
        for selector in (None, "2"):
            with (
                self.subTest(selector=selector),
                patch.object(benchmark.subprocess, "check_output", side_effect=failure),
            ):
                with self.assertRaises(ValueError):
                    benchmark.resolve_gpu(selector)
        with patch.object(benchmark.subprocess, "check_output", side_effect=[self.GPU, failure]):
            with self.assertRaises(ValueError):
                benchmark.resolve_gpu()


class SummaryContract(unittest.TestCase):
    def workload(self):
        return dict(
            warmup=1, repeats=2, cases=[dict(name="case", prompt=[1, 2], decode_tokens=[3, 4])]
        )

    def test_summary_preserves_union_of_metrics_and_arbitrary_reference(self):
        workload = self.workload()
        with tempfile.TemporaryDirectory() as directory:
            records = []
            for label in ("baseline", "candidate"):
                report = worker_report(workload)
                if label == "candidate":
                    for entry in [
                        report["cases"][0]["first_request"],
                        *report["cases"][0]["samples"],
                    ]:
                        entry["prediction_ids"] = [11, 12, 11]
                path = Path(directory) / f"{label}.json"
                write_json(path, report)
                records.append(dict(profile=label, result=str(path)))
            result = benchmark.summarize(records, workload, "baseline")
            case = result["candidate"]["cases"]["case"]
            self.assertEqual(set(benchmark.TIMING_FIELDS) - case.keys(), set())
            self.assertEqual(case["first_prediction_ms"]["n"], 2)
            self.assertEqual(case["memory"]["sampled_peak_device_delta_bytes"], 50)
            self.assertEqual(case["prediction_agreement"], dict(baseline=2))
            self.assertEqual(case["decode_tokens_per_second"], 1000 / 3)

    def test_malformed_or_inconsistent_reports_fail(self):
        workload = self.workload()
        mutations = [
            lambda r: r.update(cpu_threads=2),
            lambda r: r["cases"][0].update(prompt_tokens=1),
            lambda r: r["cases"][0].update(samples=r["cases"][0]["samples"][:1]),
            lambda r: r["cases"][0]["first_request"].update(prediction_ids=[11]),
            lambda r: r["cases"][0]["first_request"].update(consumed_positions=3),
            lambda r: r["cases"][0]["first_request"].update(prediction_ids=[12, 11, 11]),
            lambda r: r["cases"][0]["samples"][0].update(consumed_positions=3),
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "report.json"
            for mutate in mutations:
                with self.subTest(mutation=mutate):
                    report = worker_report(workload)
                    mutate(report)
                    write_json(path, report)
                    with self.assertRaises(ValueError):
                        benchmark.summarize([dict(profile="hf", result=str(path))], workload, "hf")

    def test_first_request_is_consistent_across_rounds(self):
        workload = self.workload()
        with tempfile.TemporaryDirectory() as directory:
            records = []
            for index in range(2):
                report = worker_report(workload)
                if index:
                    report["cases"][0]["first_request"]["prediction_ids"][0] = 12
                path = Path(directory) / f"{index}.json"
                write_json(path, report)
                records.append(dict(profile="hf", result=str(path)))
            with self.assertRaisesRegex(ValueError, "Repeated predictions differ"):
                benchmark.summarize(records, workload, "hf")


class ControllerContract(unittest.TestCase):
    def fixtures(self, root, model_type, mode="run"):
        model = root / "model"
        model.mkdir()
        manifest = dict(revision="fixture-revision")
        write_json(
            model / ("manifest.json" if model_type == "qwen35" else "model_manifest.json"), manifest
        )
        native = root / "xinfer_benchmark"
        native.write_bytes(b"native benchmark fixture")
        library = root / "libxinfer_attention_ops.so"
        library.write_bytes(b"attention fixture")
        directories = []
        for label in ("baseline", "candidate"):
            directory = root / label
            directory.mkdir()
            pair = metadata_pair(model_type)
            for kind, metadata in pair.items():
                package = directory / f"{kind}.pt2"
                package.write_bytes(b"compiled package fixture")
                metadata["checkpoint"]["manifest"] = manifest
                metadata.update(
                    package_bytes=package.stat().st_size, package_sha256=sha256(package)
                )
                write_json(directory / f"{kind}.json", metadata)
            directories.append(directory)
        capital = root / "reference/capital"
        capital.mkdir(parents=True)
        (capital / "prompt.ids").write_text("7 8\n")
        (capital / "generated.ids").write_text("9\n")
        args = SimpleNamespace(
            command=mode,
            gpu=None,
            warmup=1,
            repeats=1,
            rounds=2,
            decode_steps=1,
            lengths=[2],
            allow_busy=True,
            output=root / "benchmark",
            native=native,
            model_dir=model,
            reference=root / "reference",
            packages=directories[0],
            baseline_packages=directories[0],
            candidate_packages=directories[1],
            baseline_label="baseline",
            candidate_label="candidate",
            validation=root / "absent.json",
        )
        return args, library

    def run_controller(self, args, library, model_type, fail=False):
        gpu = GpuSelection.GPU
        workers, gpu_records = [], []

        def query(command, **options):
            if command[0] == "ldd":
                self.assertEqual(options["env"]["PYTHONHOME"], "/nonexistent-python-home")
                return "libtorch_cpu.so => /fixture/libtorch_cpu.so\n"
            self.assertEqual(command[:3], ["nvidia-smi", "-i", gpu])
            gpu_records.append(command)
            return f"name, uuid\nGPU 2, {gpu}\n"

        def worker(command, env, device, telemetry, log, timeout):
            workers.append(dict(command=command, env=dict(env), device=device))
            if fail and len(workers) == 2:
                raise RuntimeError("worker failed")
            workload = json.loads((args.output / "workload.json").read_text())
            write_json(Path(command[-1]), worker_report(workload))
            return 10

        weights = dict(identity_sha256="1" * 64, sources={})
        with (
            patch.dict(os.environ, CUDA_VISIBLE_DEVICES="3,1", CUDA_DEVICE_ORDER="PCI_BUS_ID"),
            patch.object(benchmark, "resolve_gpu", return_value=gpu) as resolve,
            patch.object(benchmark, "wait_idle", return_value=[]) as idle,
            patch.object(benchmark, "run_worker", side_effect=worker),
            patch.object(benchmark.platform, "platform", return_value="test-platform"),
            patch.object(benchmark.subprocess, "check_output", side_effect=query),
            patch("compiler.common.attention.native_attention_library", return_value=library),
            patch("compiler.common.weights.validate_shared_weights", return_value=weights),
            patch("transformers.AutoTokenizer.from_pretrained") as tokenizer,
            patch("compiler.qwen35.prepare.encode_prompt", return_value=[7, 8]),
        ):
            tokenizer.return_value.encode.return_value = [9]
            invoke = qwen35_benchmark.benchmark if model_type == "qwen35" else qwen3_benchmark.run
            if fail:
                with self.assertRaisesRegex(RuntimeError, "worker failed"):
                    invoke(args)
            else:
                invoke(args)
            self.assertEqual(os.environ["CUDA_VISIBLE_DEVICES"], "3,1")
            self.assertEqual(os.environ["CUDA_DEVICE_ORDER"], "PCI_BUS_ID")
        resolve.assert_called_once_with(args.gpu)
        self.assertTrue(
            all(
                row["device"] == gpu and row["env"]["CUDA_VISIBLE_DEVICES"] == gpu
                for row in workers
            )
        )
        self.assertTrue(all(call.args[0] == gpu for call in idle.call_args_list))
        self.assertEqual(len(gpu_records), 1)
        return workers, idle

    def test_models_and_compare_share_gpu_routing_rotation_and_reports(self):
        for model_type, mode in (("qwen3", "run"), ("qwen3", "compare"), ("qwen35", "run")):
            for selector in (None, "2"):
                with (
                    self.subTest(model=model_type, mode=mode, selector=selector),
                    tempfile.TemporaryDirectory() as temporary,
                ):
                    args, library = self.fixtures(Path(temporary), model_type, mode)
                    args.gpu = selector
                    workers, idle = self.run_controller(args, library, model_type)
                    self.assertEqual(len(workers), 4)
                    self.assertEqual(idle.call_count, 8)
                    is_python = [row["command"][0] == sys.executable for row in workers]
                    self.assertEqual(
                        is_python, [True, False, False, True] if mode == "run" else [False] * 4
                    )
                    if mode == "compare":
                        self.assertEqual(
                            [Path(row["command"][1]).name for row in workers],
                            ["baseline", "candidate", "candidate", "baseline"],
                        )
                    self.assertTrue(
                        all(
                            row["env"]["PYTHONHOME"] == "/nonexistent-python-home"
                            for row in workers
                            if row["command"][0] != sys.executable
                        )
                    )
                    environment = json.loads((args.output / "environment.json").read_text())
                    self.assertEqual(environment["gpu_uuid"], GpuSelection.GPU)
                    self.assertEqual(environment["attention_library"]["sha256"], sha256(library))
                    self.assertEqual(environment["binary_sha256"], sha256(args.native))
                    self.assertEqual(
                        environment["workload_sha256"], sha256(args.output / "workload.json")
                    )
                    self.assertTrue((args.output / "native.ldd").is_file())
                    summary = json.loads((args.output / "summary.json").read_text())
                    labels = (
                        {"baseline", "candidate"}
                        if mode == "compare"
                        else {"hf", "cpp_aoti"}
                        if model_type == "qwen35"
                        else {"eager", "aoti"}
                    )
                    self.assertEqual(set(summary["profiles"]), labels)
                    self.assertEqual(summary["schema_version"], 1)
                    if mode == "compare":
                        self.assertNotIn("eager_baseline", summary)
                    else:
                        self.assertEqual(summary["eager_baseline"], "hf_dynamic_cache")
                    self.assertTrue(summary["functional_checks_passed"])
                    self.assertIsNone(summary["numerical_acceptance"]["passed"])

    def test_worker_failure_persists_completed_runs_and_failure_report_for_both_models(self):
        for model_type in ("qwen3", "qwen35"):
            with self.subTest(model=model_type), tempfile.TemporaryDirectory() as temporary:
                args, library = self.fixtures(Path(temporary), model_type)
                self.run_controller(args, library, model_type, fail=True)
                self.assertEqual(len(json.loads((args.output / "runs.json").read_text())), 1)
                self.assertEqual(
                    json.loads((args.output / "failure.json").read_text()),
                    dict(error="worker failed", completed_runs=1),
                )
                self.assertFalse((args.output / "summary.json").exists())

    def test_matching_qwen35_correctness_evidence_is_preserved(self):
        with tempfile.TemporaryDirectory() as temporary:
            args, library = self.fixtures(Path(temporary), "qwen35")
            metadata = json.loads((args.packages / "decode.json").read_text())
            validation = dict(
                passed=False,
                native_bitwise=True,
                failures=[{}],
                greedy_failures=[],
                package_sha256={
                    kind: json.loads((args.packages / f"{kind}.json").read_text())["package_sha256"]
                    for kind in ("prefill", "decode")
                },
                shared_weights=metadata["shared_weights"],
            )
            args.validation = Path(temporary) / "validation.json"
            write_json(args.validation, validation)
            self.run_controller(args, library, "qwen35")
            summary = json.loads((args.output / "summary.json").read_text())
            self.assertFalse(summary["numerical_acceptance"]["passed"])
            self.assertTrue(summary["functional_checks_passed"])
            self.assertEqual(summary["numerical_acceptance"]["numerical_failures"], 1)
            self.assertEqual(json.loads((args.output / "correctness.json").read_text()), validation)

    def test_mismatched_correctness_evidence_is_rejected_before_workers(self):
        with tempfile.TemporaryDirectory() as temporary:
            args, library = self.fixtures(Path(temporary), "qwen35")
            args.validation = Path(temporary) / "validation.json"
            write_json(args.validation, dict(package_sha256={}, shared_weights={}))
            with (
                patch.object(benchmark, "resolve_gpu", return_value=GpuSelection.GPU),
                patch.object(
                    benchmark,
                    "capture_environment",
                    return_value={
                        "packages": {
                            "cpp_aoti": {
                                kind: {
                                    "metadata": json.loads(
                                        (args.packages / f"{kind}.json").read_text()
                                    )
                                }
                                for kind in ("prefill", "decode")
                            }
                        }
                    },
                ),
                patch("transformers.AutoTokenizer.from_pretrained") as tokenizer,
                patch("compiler.qwen35.prepare.encode_prompt", return_value=[7, 8]),
                patch.object(benchmark, "run_worker") as worker,
            ):
                tokenizer.return_value.encode.return_value = [9]
                with self.assertRaisesRegex(ValueError, "Validation is for different packages"):
                    qwen35_benchmark.benchmark(args)
                worker.assert_not_called()
            self.assertEqual(
                json.loads((args.output / "failure.json").read_text())["completed_runs"], 0
            )

    def test_check_idle_cli_resolves_default_and_explicit_selection(self):
        gpu = GpuSelection.GPU
        for selector in (None, "2", gpu):
            with self.subTest(selector=selector), tempfile.TemporaryDirectory() as temporary:
                output = Path(temporary) / "idle.json"
                command = ["benchmark", "check-idle", "--output", str(output)]
                if selector is not None:
                    command.extend(["--gpu", selector])
                with (
                    patch.object(sys, "argv", command),
                    patch.object(qwen35_benchmark, "resolve_gpu", return_value=gpu) as resolve,
                    patch.object(qwen35_benchmark, "wait_idle", return_value=[]) as idle,
                ):
                    qwen35_benchmark.main()
                resolve.assert_called_once_with(selector)
                idle.assert_called_once_with(gpu, output=output)

    def test_invalid_gpu_preserves_absent_output(self):
        with tempfile.TemporaryDirectory() as temporary:
            args, _ = self.fixtures(Path(temporary), "qwen35")
            args.output = Path(temporary) / "missing-parent" / "output"
            with patch.object(benchmark, "resolve_gpu", side_effect=ValueError("invalid GPU")):
                with self.assertRaisesRegex(ValueError, "invalid GPU"):
                    qwen35_benchmark.benchmark(args)
            self.assertFalse(args.output.parent.exists())


if __name__ == "__main__":
    unittest.main()
