"""CPU checks for benchmark measurement boundaries and the GPU idle guard."""

import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from compiler.common import benchmark
from compiler.qwen3 import benchmark as qwen3_benchmark
from compiler.qwen35 import benchmark as qwen35_benchmark


class BenchmarkContract(unittest.TestCase):
    def test_fixed_tokens_and_timing_boundaries(self):
        calls = []
        case = dict(prompt=[7, 8], decode_tokens=[0, 9])  # EOS-like 0 must not stop the request.

        def predict(tokens, position):
            calls.append((tokens, position))
            return 0

        with patch('torch.cuda.synchronize') as sync, \
             patch.object(benchmark.time, 'perf_counter', side_effect=[0, 1, 3, 9]):
            result = benchmark.measure_request(lambda: calls.append('reset'), predict, case)
        self.assertEqual(calls, ['reset', ([7, 8], 0), ([0], 2), ([9], 3)])
        self.assertEqual(sync.call_count, 4)
        self.assertEqual(result['prediction_ids'], [0, 0, 0])
        self.assertEqual(result['consumed_positions'], 4)
        self.assertEqual([result[k] for k in ('reset_ms', 'prefill_ms', 'first_prediction_ms',
                                             'decode_ms', 'decode_ms_per_token', 'request_ms')],
                         [1000, 2000, 3000, 6000, 3000, 9000])

    def test_first_request_and_warmup_excluded(self):
        requests, reset_after = [], []

        def request(case):
            requests.append(case)
            return dict(prediction_ids=[3], serial=len(requests))

        workload = dict(warmup=2, repeats=2, cases=[dict(name='case', prompt=[1], decode_tokens=[2])])
        with patch.object(benchmark, 'memory_stats', return_value={'device_used_bytes': 100}), \
             patch('torch.cuda.reset_peak_memory_stats', side_effect=lambda: reset_after.append(len(requests))):
            row, = benchmark.measure_cases(request, workload)
        self.assertEqual(len(requests), 5)
        self.assertEqual(row['first_request']['serial'], 1)
        self.assertEqual([s['serial'] for s in row['samples']], [4, 5])
        self.assertEqual(reset_after, [3])

    def test_idle_requires_consecutive_samples_on_same_device(self):
        readings = ['0,0', '99,0'] + ['0,0'] * 5
        with tempfile.TemporaryDirectory() as directory, \
             patch.object(benchmark.subprocess, 'check_output', side_effect=readings) as query, \
             patch.object(benchmark.time, 'sleep'):
            path = Path(directory) / 'idle.json'
            samples = benchmark.wait_idle('GPU-test', output=path)
            self.assertEqual(len(samples), 7)
            self.assertTrue(json.loads(path.read_text())['idle'])
            self.assertTrue(all(call.args[0][2] == 'GPU-test' for call in query.call_args_list))

    def test_busy_gpu_rejected_or_explicitly_provisional(self):
        with patch.object(benchmark.subprocess, 'check_output', return_value='99,90') as query, \
             patch.object(benchmark.time, 'sleep'):
            with self.assertRaisesRegex(RuntimeError, 'not idle'):
                benchmark.wait_idle('GPU-test')
            self.assertEqual(query.call_count, 30)
            self.assertEqual(len(benchmark.wait_idle('GPU-test', allow_busy=True)), 1)


class SharedWeightComparison(unittest.TestCase):
    def profiles(self):
        metadata = dict(
            model_revision='revision-one', torch_version='2.13.0+cu126', torch_cuda_version='12.6',
            batch_size=1, dtype='bfloat16', layers=28, kv_heads=8, head_dim=128, cache_length=1024,
            vocab_size=151936, cxx11_abi=True, cuda_compute_capability=[8, 9],
            cache_shape=[1, 8, 1024, 128], cache_stride=[1048576, 131072, 128, 1],
            flat_input_count=59, eos_token_ids=[151645], attention_backend='flash',
            cache_input='full_capacity_with_extent',
            custom_attention=dict(name='xinfer::flash_attention', abi_version=1, read_only=True),
            valid_cache_length=[1, 1024],
            kv_extent=dict(device='cpu', dtype='uint8', shape='[valid_length + 1]', data_read=False),
            export_options=dict(backed_size_oblivious=False), triton_enable_fp_fusion=True,
            inductor_configs={'emulate_precision_casts': True,
                              'aot_inductor.package_constants_in_so': False,
                              'always_keep_tensor_constants': True},
            shared_weights=dict(format=2, file='weights.pt', identity='1' * 64,
                                bytes=1000000, sha256='a' * 64))
        baseline = {kind: dict(metadata=dict(copy.deepcopy(metadata), entry_point=kind,
                                            token_length=[2, 512] if kind == 'prefill' else [1, 1]))
                    for kind in ('prefill', 'decode')}
        candidate = copy.deepcopy(baseline)
        for entry in candidate.values():
            entry['metadata']['shared_weights'].update(bytes=1000128, sha256='b' * 64)
        weights = dict(identity_sha256='1' * 64,
                       sources={'model.model.embed_tokens.weight': 'v_' + 'c' * 64,
                                'model.model.layers.0.self_attn.q_proj.weight': 'v_' + 'd' * 64},
                       entries={'decode': {'constants': {'_tensor_constant0': 'v_generated_a'}}})
        other_weights = copy.deepcopy(weights)
        other_weights['entries']['decode']['constants']['_tensor_constant0'] = 'v_generated_b'
        return baseline, candidate, weights, other_weights

    def test_same_weights_with_different_archive_hashes_are_comparable(self):
        baseline, candidate, before, after = self.profiles()
        qwen3_benchmark.validate_packages(baseline)
        qwen3_benchmark.validate_packages(candidate)
        qwen3_benchmark.validate_comparison(baseline, candidate, before, after)
        candidate['prefill']['metadata']['shared_weights']['sha256'] = 'e' * 64
        with self.assertRaisesRegex(ValueError, 'Package metadata differs: shared_weights'):
            qwen3_benchmark.validate_packages(candidate)

    def test_changed_source_weights_are_rejected_despite_same_identity(self):
        baseline, candidate, before, after = self.profiles()
        after['sources']['model.model.embed_tokens.weight'] = 'v_' + 'e' * 64
        with self.assertRaisesRegex(ValueError, 'Incomparable shared weights: sources'):
            qwen3_benchmark.validate_comparison(baseline, candidate, before, after)

    def test_changed_numerical_settings_are_rejected(self):
        baseline, candidate, before, after = self.profiles()
        for entry in candidate.values():
            entry['metadata']['triton_enable_fp_fusion'] = False
        qwen3_benchmark.validate_packages(candidate)
        with self.assertRaisesRegex(ValueError, 'Incomparable packages: prefill/triton_enable_fp_fusion'):
            qwen3_benchmark.validate_comparison(baseline, candidate, before, after)


class GpuSelection(unittest.TestCase):
    GPU = 'GPU-22222222-2222-2222-2222-222222222222'
    OTHER = 'GPU-33333333-3333-3333-3333-333333333333'

    def test_both_models_use_the_shared_resolver(self):
        self.assertIs(qwen3_benchmark.resolve_gpu, benchmark.resolve_gpu)
        self.assertIs(qwen35_benchmark.resolve_gpu, benchmark.resolve_gpu)

    def test_explicit_index_or_uuid_overrides_inherited_visibility(self):
        for selector in ('2', self.GPU):
            with self.subTest(selector=selector), \
                    patch.dict(os.environ, CUDA_VISIBLE_DEVICES='0'), \
                    patch.object(benchmark.subprocess, 'check_output', return_value=self.GPU + '\n') as query:
                self.assertEqual(benchmark.resolve_gpu(selector), self.GPU)
                query.assert_called_once()
                self.assertEqual(query.call_args.args[0][:3], ['nvidia-smi', '-i', selector])
                self.assertIn('--query-gpu=uuid', query.call_args.args[0])
                self.assertEqual(os.environ['CUDA_VISIBLE_DEVICES'], '0')

    def test_default_respects_cuda_visibility_and_order_without_parent_cuda(self):
        calls = []

        def query(command, **options):
            calls.append(command)
            if command[0] == sys.executable:
                self.assertEqual(command[1], '-c')
                child_env = options.get('env', os.environ)
                self.assertEqual(child_env['CUDA_VISIBLE_DEVICES'], '3,1')
                self.assertEqual(child_env['CUDA_DEVICE_ORDER'], 'PCI_BUS_ID')
            else:
                self.assertEqual(command[:3], ['nvidia-smi', '-i', self.GPU])
            return self.GPU + '\n'

        with patch.dict(os.environ, CUDA_VISIBLE_DEVICES='3,1', CUDA_DEVICE_ORDER='PCI_BUS_ID'), \
                patch.object(benchmark.subprocess, 'check_output', side_effect=query), \
                patch('torch.cuda.get_device_properties', side_effect=AssertionError('Parent must not initialize CUDA')):
            self.assertEqual(benchmark.resolve_gpu(), self.GPU)
            self.assertEqual(os.environ['CUDA_VISIBLE_DEVICES'], '3,1')
            self.assertEqual(os.environ['CUDA_DEVICE_ORDER'], 'PCI_BUS_ID')
        self.assertEqual(len(calls), 2)

    def test_invalid_explicit_selections_do_not_probe_or_fall_back(self):
        for selector in ('', ' ', '0,1', 'MIG-example'):
            with self.subTest(selector=selector), \
                    patch.object(benchmark.subprocess, 'check_output') as query:
                with self.assertRaises(ValueError):
                    benchmark.resolve_gpu(selector)
                query.assert_not_called()

    def test_empty_multiple_or_nonphysical_uuid_results_are_rejected(self):
        for result in ('', self.GPU + '\n' + self.OTHER, 'MIG-example', 'not-a-gpu'):
            for selector in (None, '2'):
                with self.subTest(result=result, selector=selector), \
                        patch.object(benchmark.subprocess, 'check_output', return_value=result) as query:
                    with self.assertRaises(ValueError):
                        benchmark.resolve_gpu(selector)
                    query.assert_called_once()

    def test_cuda_uuid_cannot_silently_resolve_to_another_physical_gpu(self):
        with patch.object(benchmark.subprocess, 'check_output',
                          side_effect=[self.GPU, self.OTHER]) as query:
            with self.assertRaisesRegex(ValueError, 'UUID does not match'):
                benchmark.resolve_gpu()
        self.assertEqual(query.call_count, 2)

    def test_missing_cuda_or_failed_smi_lookup_never_falls_back(self):
        failure = subprocess.CalledProcessError(1, ['device-query'], stderr='No accessible CUDA device')
        for selector in (None, '2'):
            with self.subTest(selector=selector), \
                    patch.dict(os.environ, CUDA_VISIBLE_DEVICES=''), \
                    patch.object(benchmark.subprocess, 'check_output', side_effect=failure) as query:
                with self.assertRaises(ValueError):
                    benchmark.resolve_gpu(selector)
                query.assert_called_once()
        with patch.object(benchmark.subprocess, 'check_output',
                          side_effect=[self.GPU, failure]) as query:
            with self.assertRaises(ValueError):
                benchmark.resolve_gpu()
        self.assertEqual(query.call_count, 2)

    def test_idle_wrapper_forwards_selected_device(self):
        selected = []

        def idle(device='0', allow_busy=False):
            selected.append((device, allow_busy))
            return ['reading']

        with patch.object(qwen3_benchmark, 'check_idle', side_effect=idle):
            self.assertEqual(qwen3_benchmark.wait_idle(self.GPU, allow_busy=True), ['reading'])
        self.assertEqual(selected, [(self.GPU, True)])


class GpuBenchmarkRouting(unittest.TestCase):
    def test_run_and_compare_use_one_gpu_for_workers_idle_telemetry_and_metadata(self):
        gpu = GpuSelection.GPU
        profiles, _, weights, _ = SharedWeightComparison().profiles()
        for mode in ('run', 'compare'):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                capital = root / 'reference/capital'
                capital.mkdir(parents=True)
                (capital / 'prompt.ids').write_text('7 8\n')
                (capital / 'generated.ids').write_text('9\n')
                model = root / 'model'
                model.mkdir()
                (model / 'model_manifest.json').write_text('{}')
                native = root / 'qwen_benchmark'
                native.write_bytes(b'native benchmark fixture')
                library = root / 'attention.so'
                library.write_bytes(b'attention fixture')
                directories = []
                for label in ('baseline', 'candidate'):
                    directory = root / label
                    directory.mkdir()
                    for kind, entry in profiles.items():
                        (directory / f'{kind}.json').write_text(json.dumps(entry['metadata']))
                        (directory / f'{kind}.pt2').write_bytes(b'compiled package fixture')
                    directories.append(directory)
                args = SimpleNamespace(
                    command=mode, gpu=None, warmup=1, repeats=1, rounds=2, decode_steps=1, lengths=[2],
                    allow_busy=True, output=root / 'benchmark', native=native, model_dir=model,
                    reference=root / 'reference', packages=directories[0],
                    baseline_packages=directories[0], candidate_packages=directories[1],
                    baseline_label='baseline', candidate_label='candidate')
                workers, gpu_records = [], []

                def query(command, **options):
                    if command[0] == 'git':
                        return 'fixture-commit\n'
                    self.assertEqual(command[:3], ['nvidia-smi', '-i', gpu])
                    gpu_records.append(command)
                    return f'name, uuid\nGPU 2, {gpu}\n'

                def worker(command, env, device, telemetry, log, timeout):
                    workers.append(dict(command=command, env=dict(env), device=device))
                    case = dict(name='length_2', memory=dict(
                        peak_allocated_bytes=0, peak_reserved_bytes=0, sampled_peak_device_used_bytes=100),
                        samples=[dict(reset_ms=1, prefill_ms=2, decode_ms_per_token=3,
                                      request_ms=5, prediction_ids=[11, 12])])
                    Path(command[-1]).write_text(json.dumps(dict(
                        load_ms=1, initial_memory=dict(device_used_bytes=100), cases=[case])))
                    return 10

                with patch.dict(os.environ, CUDA_VISIBLE_DEVICES='3,1', CUDA_DEVICE_ORDER='PCI_BUS_ID'), \
                        patch.object(qwen3_benchmark, 'resolve_gpu', return_value=gpu) as resolve, \
                        patch.object(qwen3_benchmark, 'wait_idle', return_value=[]) as idle, \
                        patch.object(qwen3_benchmark, 'run_worker', side_effect=worker), \
                        patch.object(qwen3_benchmark.platform, 'platform', return_value='test-platform'), \
                        patch.object(qwen3_benchmark.subprocess, 'check_output', side_effect=query), \
                        patch('compiler.qwen3.custom_ops.attention.native_attention_library', return_value=library), \
                        patch('compiler.common.weights.validate_shared_weights', return_value=weights):
                    qwen3_benchmark.run(args)
                    self.assertEqual(os.environ['CUDA_VISIBLE_DEVICES'], '3,1')
                    self.assertEqual(os.environ['CUDA_DEVICE_ORDER'], 'PCI_BUS_ID')
                resolve.assert_called_once_with(None)
                self.assertEqual(len(workers), 4)
                self.assertTrue(all(call['device'] == gpu for call in workers))
                self.assertTrue(all(call['env']['CUDA_VISIBLE_DEVICES'] == gpu for call in workers))
                self.assertEqual(idle.call_count, 8)
                for call in idle.call_args_list:
                    self.assertEqual(call.args[0] if call.args else call.kwargs['device'], gpu)
                self.assertEqual(len(gpu_records), 1)
                environment = json.loads((args.output / 'environment.json').read_text())
                self.assertEqual(environment['gpu_uuid'], gpu)
                self.assertIn(gpu, environment['gpu'])
                summary = json.loads((args.output / 'summary.json').read_text())
                self.assertEqual(set(summary['profiles']), {'eager', 'aoti'} if mode == 'run'
                                 else {'baseline', 'candidate'})
                if mode == 'run':
                    self.assertEqual(sum(call['command'][0] == sys.executable for call in workers), 2)
                else:
                    self.assertTrue(all(call['command'][0] == str(native.resolve()) for call in workers))


class Qwen35GpuRouting(unittest.TestCase):
    def test_run_routes_workers_idle_telemetry_and_metadata_to_selected_gpu(self):
        gpu = GpuSelection.GPU
        for selector in (None, '2'):
            with self.subTest(selector=selector), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                model = root / 'model'
                model.mkdir()
                manifest = dict(revision='fixture-revision')
                (model / 'manifest.json').write_text(json.dumps(manifest))
                native = root / 'qwen35_benchmark'
                native.write_bytes(b'native benchmark fixture')
                packages = root / 'packages'
                packages.mkdir()
                for kind in ('prefill', 'decode'):
                    package = packages / f'{kind}.pt2'
                    package.write_bytes(b'compiled package fixture')
                    metadata = dict(checkpoint=manifest, package_sha256=qwen35_benchmark.sha256(package))
                    (packages / f'{kind}.json').write_text(json.dumps(metadata))
                args = SimpleNamespace(
                    gpu=selector, warmup=1, repeats=1, rounds=2, decode_steps=1, lengths=[2],
                    allow_busy=True, output=root / 'benchmark', native=native, model_dir=model,
                    packages=packages, validation=root / 'no-validation.json')
                workers, gpu_records = [], []

                def query(command, **options):
                    if command[0] == 'git':
                        return 'fixture-commit\n'
                    if command[0] == 'ldd':
                        return 'libtorch_cpu.so => /fixture/libtorch_cpu.so\n'
                    self.assertEqual(command[:3], ['nvidia-smi', '-i', gpu])
                    gpu_records.append(command)
                    return f'name, uuid\nGPU 2, {gpu}\n'

                def worker(command, env, device, telemetry, log, timeout):
                    workers.append(dict(command=command, env=dict(env), device=device))
                    sample = dict(reset_ms=1, prefill_ms=2, first_prediction_ms=3,
                                  decode_ms_per_token=3, request_ms=6,
                                  prediction_ids=[11, 12], consumed_positions=3)
                    case = dict(name='length_2', first_request=sample, samples=[sample], memory=dict(
                        peak_allocated_bytes=0, peak_reserved_bytes=0, sampled_peak_device_used_bytes=100))
                    Path(command[-1]).write_text(json.dumps(dict(cpu_threads=1, load_ms=1, cases=[case])))
                    return 10

                with patch.dict(os.environ, CUDA_VISIBLE_DEVICES='3,1', CUDA_DEVICE_ORDER='PCI_BUS_ID'), \
                        patch.object(qwen35_benchmark, 'resolve_gpu', return_value=gpu) as resolve, \
                        patch.object(qwen35_benchmark, 'wait_idle', return_value=[]) as idle, \
                        patch.object(qwen35_benchmark, 'run_worker', side_effect=worker), \
                        patch.object(qwen35_benchmark.platform, 'platform', return_value='test-platform'), \
                        patch.object(qwen35_benchmark.subprocess, 'check_output', side_effect=query), \
                        patch('transformers.AutoTokenizer.from_pretrained') as tokenizer, \
                        patch('compiler.qwen35.prepare.encode_prompt', return_value=[7, 8]), \
                        patch('compiler.common.weights.validate_shared_weights', return_value={}):
                    tokenizer.return_value.encode.return_value = [9]
                    qwen35_benchmark.benchmark(args)
                    self.assertEqual(os.environ['CUDA_VISIBLE_DEVICES'], '3,1')
                    self.assertEqual(os.environ['CUDA_DEVICE_ORDER'], 'PCI_BUS_ID')
                resolve.assert_called_once_with(selector)
                self.assertEqual(len(workers), 4)
                self.assertTrue(all(call['device'] == gpu for call in workers))
                self.assertTrue(all(call['env']['CUDA_VISIBLE_DEVICES'] == gpu for call in workers))
                self.assertEqual(sum(call['command'][0] == sys.executable for call in workers), 2)
                self.assertEqual(idle.call_count, 8)
                self.assertTrue(all(call.args[1] == gpu for call in idle.call_args_list))
                self.assertEqual(len(gpu_records), 1)
                environment = json.loads((args.output / 'environment.json').read_text())
                self.assertEqual(environment['gpu_uuid'], gpu)
                self.assertIn(gpu, environment['gpu'])
                summary = json.loads((args.output / 'summary.json').read_text())
                self.assertEqual(set(summary['profiles']), {'hf', 'cpp_aoti'})
                self.assertTrue(summary['functional_checks_passed'])

    def test_check_idle_cli_resolves_default_and_explicit_selection(self):
        gpu = GpuSelection.GPU
        for selector in (None, '2', gpu):
            with self.subTest(selector=selector), tempfile.TemporaryDirectory() as temporary:
                output = Path(temporary) / 'idle.json'
                command = ['benchmark', 'check-idle', '--output', str(output)]
                if selector is not None:
                    command.extend(['--gpu', selector])
                with patch.object(sys, 'argv', command), \
                        patch.object(qwen35_benchmark, 'resolve_gpu', return_value=gpu) as resolve, \
                        patch.object(qwen35_benchmark, 'wait_idle', return_value=[]) as idle:
                    qwen35_benchmark.main()
                resolve.assert_called_once_with(selector)
                idle.assert_called_once_with(output, gpu)

    def test_invalid_selection_preserves_absent_run_and_idle_outputs(self):
        for mode in ('run', 'check-idle'):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as temporary:
                output = Path(temporary) / 'missing-parent' / 'output'
                with patch.object(qwen35_benchmark, 'resolve_gpu', side_effect=ValueError('invalid GPU')):
                    with self.assertRaisesRegex(ValueError, 'invalid GPU'):
                        if mode == 'run':
                            args = SimpleNamespace(gpu='bad', warmup=1, repeats=1, rounds=1,
                                                   decode_steps=1, lengths=[2], output=output)
                            qwen35_benchmark.benchmark(args)
                        else:
                            with patch.object(sys, 'argv', ['benchmark', 'check-idle', '--gpu', 'bad',
                                                            '--output', str(output)]):
                                qwen35_benchmark.main()
                self.assertFalse(output.parent.exists())


if __name__ == '__main__':
    unittest.main()
