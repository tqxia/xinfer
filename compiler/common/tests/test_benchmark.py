"""CPU checks for benchmark measurement boundaries and the GPU idle guard."""

import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from compiler.common import benchmark
from compiler.qwen3 import benchmark as qwen3_benchmark


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


if __name__ == '__main__':
    unittest.main()
