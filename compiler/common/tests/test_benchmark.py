"""CPU checks for benchmark measurement boundaries and the GPU idle guard."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from compiler.common import benchmark


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


if __name__ == '__main__':
    unittest.main()
