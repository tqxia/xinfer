"""CPU checks for validation-stage coverage, policies and request orchestration."""

from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock, patch
import unittest

import torch

from tools.validation.common.chain import Policy, compare_traces, new_report
from tools.validation.common.runner import (
    Stage,
    Trace,
    capture_state,
    greedy,
    run_stage,
    teacher_trace,
)
from tools.validation.common.pipeline import _greedy_anchors, validate_chain


CONV = dict(layer=0, kind="conv", shape=[1, 2, 4], dtype="float32", update="replace")
KEY = dict(layer=0, kind="key", shape=[1, 1, 8, 1], dtype="float32", update="inplace")
CASE_KEY = ("case", 0)


class StageComparisons(unittest.TestCase):
    def trace(self, logits=None, states=None, generated=None):
        return Trace(
            logits={CASE_KEY: torch.ones(1, 2)} if logits is None else logits,
            states={} if states is None else states,
            greedy={} if generated is None else generated,
        )

    def compare(self, actual, expected, *, policy=None, specs=(), **kwargs):
        report = new_report("qwen35")
        compare_traces(
            report,
            "python_aoti",
            "hf",
            actual,
            expected,
            specs,
            policy or Policy(),
            **kwargs,
        )
        return report

    def test_missing_or_extra_logit_steps_cannot_pass(self):
        expected = self.trace()
        for logits in ({}, {CASE_KEY: torch.ones(1, 2), ("extra", 0): torch.ones(1, 2)}):
            with self.subTest(keys=list(logits)), self.assertRaises(ValueError):
                self.compare(self.trace(logits=logits), expected)

    def test_missing_state_steps_and_tensor_counts_cannot_pass(self):
        expected = self.trace(states={CASE_KEY: (torch.ones(1, 2, 4),)})
        for states in ({}, {CASE_KEY: ()}, {CASE_KEY: (torch.ones(1, 2, 4),) * 2}):
            with self.subTest(states=states), self.assertRaises(ValueError):
                self.compare(self.trace(states=states), expected, specs=[CONV])

    def test_explicit_scopes_allow_partial_capture_but_require_every_selected_step(self):
        first, last = ("case", 0), ("case", 1)
        actual = self.trace(logits={first: torch.ones(1, 2), last: torch.ones(1, 2)})
        expected = self.trace(logits={first: torch.ones(1, 2)})
        report = self.compare(actual, expected, logit_keys={first}, state_keys=set())
        self.assertEqual(report["failures"], [])
        with self.assertRaises(ValueError):
            self.compare(actual, expected, logit_keys={last}, state_keys=set())

    def test_invalid_shapes_and_nonfinite_values_fail_even_for_diagnostics(self):
        for logits in (
            torch.ones(2, 1),
            torch.tensor([[float("nan"), 1.0]]),
            torch.tensor([[float("inf"), 1.0]]),
        ):
            with self.subTest(logits=logits), self.assertRaises(ValueError):
                self.compare(self.trace(logits={CASE_KEY: logits}), self.trace(), required=False)

    def test_native_parity_preserves_signed_zero_and_dtype(self):
        reference = self.trace(logits={CASE_KEY: torch.tensor([[0.0, 1.0]])})
        for tensor in (torch.tensor([[-0.0, 1.0]]), torch.tensor([[0.0, 1.0]]).double()):
            actual = self.trace(logits={CASE_KEY: tensor})
            with self.subTest(dtype=tensor.dtype, signed_zero=bool(torch.signbit(tensor[0, 0]))):
                numerical = self.compare(actual, reference)
                self.assertEqual(numerical["failures"], [])
                exact = self.compare(actual, reference, exact=True)
                self.assertTrue(exact["failures"])

    def test_hf_anchor_detects_drift_that_passes_both_adjacent_edges(self):
        hf = self.trace()
        eager = self.trace(logits={CASE_KEY: torch.full((1, 2), 1.04)})
        compiled = self.trace(logits={CASE_KEY: torch.full((1, 2), 1.04**2)})
        report = new_report("qwen3")
        for actual_name, expected_name, actual, expected in (
            ("eager", "hf", eager, hf),
            ("python_aoti", "eager", compiled, eager),
        ):
            compare_traces(report, actual_name, expected_name, actual, expected, [], Policy())
        self.assertEqual(report["failures"], [])
        compare_traces(report, "python_aoti", "hf", compiled, hf, [], Policy())
        self.assertTrue(report["failures"])
        self.assertTrue(all(row["comparison"] == "python_aoti_vs_hf" for row in report["failures"]))

    def test_state_acceptance_is_model_policy(self):
        reference = self.trace(states={CASE_KEY: (torch.ones(1, 2, 4),)})
        actual = self.trace(states={CASE_KEY: (torch.full((1, 2, 4), 1.06),)})
        diagnostic = self.compare(
            actual, reference, specs=[CONV], policy=Policy(state_required=False)
        )
        self.assertEqual(diagnostic["failures"], [])
        self.assertTrue(diagnostic["diagnostics"])
        required = self.compare(actual, reference, specs=[CONV], policy=Policy(state_required=True))
        self.assertTrue(required["failures"])
        self.assertEqual(required["failures"][0]["tensor"], "layer_0.conv")

    def test_eager_anchor_detects_compilation_drift_that_cancels_eager_error(self):
        hf = self.trace()
        eager = self.trace(logits={CASE_KEY: torch.full((1, 2), 1.08)})
        compiled = self.trace()
        report = new_report("qwen35")
        compare_traces(report, "python_aoti", "hf", compiled, hf, [], Policy())
        self.assertEqual(report["failures"], [])
        compare_traces(report, "python_aoti", "eager", compiled, eager, [], Policy())
        self.assertTrue(report["failures"])
        self.assertTrue(
            all(row["comparison"] == "python_aoti_vs_eager" for row in report["failures"])
        )

    def test_top1_drift_is_independent_of_bounded_logit_error(self):
        reference = self.trace(logits={CASE_KEY: torch.tensor([[1.0, 0.99]])})
        actual = self.trace(logits={CASE_KEY: torch.tensor([[0.99, 1.0]])})
        diagnostic = self.compare(actual, reference, policy=Policy(tokens_required=False))
        self.assertEqual(diagnostic["failures"], [])
        self.assertTrue(diagnostic["top1_differences"])
        required = self.compare(actual, reference, policy=Policy(tokens_required=True))
        self.assertTrue(required["failures"])
        self.assertTrue(required["top1_differences"])

    def test_token_gate_can_select_end_to_end_checkpoints(self):
        model, end = ("model_case", 0), ("end_to_end_case", 0)
        expected = self.trace(
            logits={model: torch.tensor([[1.0, 0.99]]), end: torch.tensor([[1.0, 0.99]])}
        )
        actual = self.trace(logits={key: value.flip(-1) for key, value in expected.logits.items()})
        report = self.compare(
            actual, expected, policy=Policy(tokens_required=True, token_keys={end})
        )
        self.assertEqual([row["case"] for row in report["failures"]], [end[0]])
        self.assertEqual(len(report["top1_differences"]), 2)

    def test_diagnostics_do_not_inherit_required_state_or_token_gates(self):
        reference = self.trace(
            logits={CASE_KEY: torch.tensor([[1.0, 0.99]])},
            states={CASE_KEY: (torch.ones(1, 2, 4),)},
        )
        actual = self.trace(
            logits={CASE_KEY: torch.tensor([[0.99, 1.0]])},
            states={CASE_KEY: (torch.full((1, 2, 4), 1.2),)},
        )
        report = self.compare(
            actual,
            reference,
            specs=[CONV],
            policy=Policy(state_required=True, tokens_required=True),
            required=False,
        )
        self.assertEqual(report["failures"], [])
        self.assertTrue(report["diagnostics"])
        self.assertTrue(report["top1_differences"])

    def generated(self, ids):
        return dict(
            ids=ids,
            rows=[torch.tensor([[1.0, 0.99]]), torch.tensor([[0.99, 1.0]])],
            state=(),
            position=3,
        )

    def test_greedy_divergence_uses_token_policy(self):
        reference = self.trace(generated={"chat": self.generated([0, 1])})
        actual = self.trace(generated={"chat": self.generated([0, 0])})
        diagnostic = self.compare(actual, reference, policy=Policy(tokens_required=False))
        self.assertEqual(diagnostic["failures"], [])
        self.assertTrue(diagnostic["greedy_differences"])
        required = self.compare(actual, reference, policy=Policy(tokens_required=True))
        self.assertTrue(required["failures"])
        self.assertTrue(required["greedy_failures"])
        diagnostic = self.compare(
            actual, reference, policy=Policy(tokens_required=True), required=False
        )
        self.assertEqual(diagnostic["failures"], [])
        self.assertEqual(diagnostic["greedy_failures"], [])


class CPURuntime:
    """Small stateful engine exposing the same request protocol as real stages."""

    def __init__(self, *, choices=None, wrong_position=False, corrupt_future=False):
        self.choices = choices
        self.wrong_position = wrong_position
        self.corrupt_future = corrupt_future
        self.calls = []
        self.resets = []

    def reset(self, fill=0):
        self.position = 0
        self.choice = 0
        self.state = [torch.full((1, 1, 8, 1), float(fill)), torch.zeros(1, 2, 4)]
        self.resets.append(fill)

    def step(self, tokens):
        start, end = self.position, self.position + len(tokens)
        if end > 8:
            raise ValueError("Context capacity exceeded")
        self.calls.append(tuple(tokens))
        self.state[0][0, 0, start:end, 0] = torch.tensor(tokens)
        self.state[1] = self.state[1] + sum(tokens)
        if self.corrupt_future:
            self.state[0][:, :, end:] += 1
        self.position = end + int(self.wrong_position)
        if self.choices is not None:
            token = self.choices[min(self.choice, len(self.choices) - 1)]
            self.choice += 1
            scores = torch.zeros(1, 3)
            scores[0, token] = 1
            return scores
        total = self.state[0][:, :, :end].sum()
        return torch.stack((total, total + 1, total - 1)).view(1, 3)


class RequestRuntime:
    """Record graph calls without model work, including state before rejection."""

    def __init__(self):
        self.capacity, self.max_prefill, self.vocab_size = 16384, 512, 3
        self.position, self.state, self.calls, self.resets = 17, [], [], []

    def reset(self, fill=0):
        self.position = 0
        self.resets.append(fill)

    def step(self, tokens):
        if not 1 <= len(tokens) <= self.max_prefill:
            raise ValueError("Invalid compiled call")
        self.calls.append(tuple(tokens))
        self.position += len(tokens)
        return torch.tensor([[0.0, 1.0, 0.0]], dtype=torch.bfloat16)


class ValidationPipeline(unittest.TestCase):
    def test_chain_completes_without_fp32_hooks_or_report_fields(self):
        from tools.validation.common import pipeline

        cases = [dict(name="teacher", steps=[[1, 2], [1]])]
        chats = [dict(name="chat", prompt=[1, 2], limit=2)]
        specs = [KEY, CONV]
        capture = lambda case, step: True

        class BF16Runtime(CPURuntime):
            def step(self, tokens):
                return super().step(tokens).to(torch.bfloat16)

            def close(self):
                pass

        def stages():
            for name in ("hf", "hf_dynamic", "eager"):
                yield Stage(name, BF16Runtime())

        for model_type in ("qwen3", "qwen35"):
            with self.subTest(model_type=model_type):
                suite = SimpleNamespace(
                    model_type=model_type,
                    specs=specs,
                    capacity=8,
                    state_required=model_type == "qwen35",
                    tokens_required=model_type == "qwen35",
                    eos=[2],
                    stages=stages,
                    fixtures=lambda hf: (cases, chats),
                    stage_cases=lambda name, cases: cases,
                    capture_states=capture,
                    native_state_keys=lambda cases: {("teacher", 0), ("teacher", 1)},
                    reference=Mock(side_effect=AssertionError("Unexpected HF reload")),
                )
                native = run_stage(BF16Runtime(), cases, chats, specs, capture, suite.eos)
                report = new_report(model_type)
                with (
                    patch.object(pipeline, "sha256", return_value="native-digest"),
                    patch.object(pipeline, "load_aoti_runtime", return_value=BF16Runtime()),
                    patch.object(pipeline, "_contract_probes", return_value={}),
                    patch.object(pipeline, "run_native", return_value=(native, {})),
                    patch.object(pipeline.torch.cuda, "empty_cache"),
                ):
                    validate_chain(suite, "packages", "native", "output", {}, report)
                self.assertTrue(report["passed"])
                self.assertTrue(report["native_bitwise"])
                self.assertEqual(
                    [stage["name"] for stage in report["stages"]],
                    ["hf", "hf_dynamic", "eager", "python_aoti", "cpp_aoti"],
                )
                self.assertEqual(report["reference_protocol"]["dtype"], "bfloat16")
                self.assertNotIn("fp32_controls", report["numerical_policy"])
                self.assertEqual(report["controls"], dict(matched_greedy_logits=True))
                self.assertFalse(
                    any("hf_fp32" in row["comparison"] for row in report["comparisons"])
                )
                suite.reference.assert_not_called()


class StageExecution(unittest.TestCase):
    specs = [KEY, CONV]

    def cases(self):
        return [
            dict(name="clean", steps=[[1, 2], [3]]),
            dict(name="poison", steps=[[1, 2], [3]], poison_kv=7, same_as="clean"),
        ]

    def test_teacher_stages_reset_and_consume_identical_partitions(self):
        first, second = CPURuntime(), CPURuntime()
        capture = lambda case, step: True
        left = teacher_trace(first, self.cases(), self.specs, capture)
        right = teacher_trace(second, self.cases(), self.specs, capture)
        self.assertEqual(first.resets, [0, 7])
        self.assertEqual(first.calls, [(1, 2), (3,), (1, 2), (3,)])
        self.assertEqual(first.calls, second.calls)
        self.assertEqual(left.logits.keys(), right.logits.keys())
        self.assertEqual(len(left.logits), 4)
        self.assertTrue(torch.equal(left.logits["clean", 1], left.logits["poison", 1]))
        self.assertTrue(bool((left.states["poison", 1][0][:, :, 3:] == 7).all()))

    def test_capture_scope_and_snapshots_do_not_alias_live_state(self):
        runtime = CPURuntime()
        trace = teacher_trace(
            runtime,
            [self.cases()[0]],
            self.specs,
            lambda case, step: step == 0,
        )
        self.assertEqual(set(trace.states), {("clean", 0)})
        self.assertEqual(trace.states["clean", 0][0][0, 0, 2, 0].item(), 0)
        before = trace.states["clean", 0][0].clone()
        runtime.state[0].fill_(-99)
        self.assertTrue(torch.equal(trace.states["clean", 0][0], before))

    def test_poison_checks_support_selected_state_capture(self):
        trace = teacher_trace(
            CPURuntime(),
            self.cases(),
            self.specs,
            lambda case, step: step == len(case["steps"]) - 1,
        )
        self.assertEqual(set(trace.states), {("clean", 1), ("poison", 1)})
        for step in (0, 1):
            self.assertTrue(torch.equal(trace.logits["clean", step], trace.logits["poison", step]))

    def test_wrong_position_and_poisoned_future_mutation_are_rejected(self):
        for runtime in (CPURuntime(wrong_position=True), CPURuntime(corrupt_future=True)):
            with self.subTest(runtime=runtime.__dict__), self.assertRaises(ValueError):
                teacher_trace(runtime, self.cases(), self.specs, lambda case, step: True)

    def test_greedy_stops_at_eos_without_feeding_eos_back(self):
        runtime = CPURuntime(choices=[1, 2])
        result = greedy(runtime, [0, 1], 5, [2])
        self.assertEqual(result["ids"], [1, 2])
        self.assertEqual(len(result["rows"]), 2)
        self.assertEqual(runtime.calls, [(0, 1), (1,)])
        self.assertEqual(result["position"], 3)

    def test_greedy_budget_one_does_not_decode_an_unused_token(self):
        runtime = CPURuntime(choices=[1, 2])
        result = greedy(runtime, [0, 1], 1, [2])
        self.assertEqual(result["ids"], [1])
        self.assertEqual(runtime.calls, [(0, 1)])
        self.assertEqual(result["position"], 2)

    def test_uncapped_requests_above_8k_accept_an_exact_context_budget(self):
        runtime = RequestRuntime()
        result = greedy(runtime, [0] * 8193, 8191, [1])
        self.assertEqual(result["ids"], [1])
        self.assertEqual(result["position"], 8193)
        self.assertEqual(list(map(len, runtime.calls)), [512] * 16 + [1])

    def test_requested_budget_rejects_before_mutation_without_clamping(self):
        runtime = RequestRuntime()
        with self.assertRaisesRegex(ValueError, "Generation budget exceeds context"):
            greedy(runtime, [0] * 8193, 8192, [1])
        self.assertEqual(runtime.position, 17)
        self.assertEqual(runtime.resets, [])
        self.assertEqual(runtime.calls, [])

    def test_long_prompt_consumes_every_chunk_before_selecting_a_token(self):
        runtime = CPURuntime(choices=[2, 2, 1])
        runtime.max_prefill = 2
        result = greedy(runtime, [0, 1, 0, 1, 0], 1, [2])
        self.assertEqual(runtime.calls, [(0, 1), (0, 1), (0,)])
        self.assertEqual(result["ids"], [1])
        self.assertEqual(result["position"], 5)

    def test_greedy_anchor_replays_long_prompt_with_matching_partitions(self):
        runtime = CPURuntime(choices=[0, 0, 1, 2])
        runtime.max_prefill = 2
        suite = SimpleNamespace(reference=Mock(return_value=nullcontext(runtime)))
        hf = Trace(greedy={"chat": dict(ids=[0, 0])})
        compiled = Trace(
            greedy={
                "chat": dict(
                    ids=[1, 2],
                    rows=[torch.tensor([[0.0, 1.0, 0.0]]), torch.tensor([[0.0, 0.0, 1.0]])],
                )
            }
        )
        report = new_report("qwen35")
        _greedy_anchors(
            suite,
            report,
            hf,
            compiled,
            self.specs,
            Policy(),
            [dict(name="chat", prompt=[0, 1, 0, 1, 0])],
        )
        self.assertEqual(runtime.calls, [(0, 1), (0, 1), (0,), (1,)])
        suite.reference.assert_called_once_with()
        self.assertEqual(report["failures"], [])
        self.assertEqual(len(report["comparisons"]), 2)

    def test_compact_snapshot_checks_future_and_owns_valid_prefix(self):
        runtime = CPURuntime()
        runtime.reset()
        runtime.capacity = 2048
        runtime.position = 2
        runtime.state = [torch.zeros(1, 1, 2048, 1), torch.ones(1, 2, 4)]
        specs = [dict(KEY, shape=[1, 1, 2048, 1]), CONV]
        runtime.state[0][:, :, :2] = 3
        saved = capture_state(runtime, specs)
        self.assertEqual(saved[0].shape, (1, 1, 2, 1))
        self.assertEqual(saved[1].shape, (1, 2, 4))
        runtime.state[0][:, :, :2] = 9
        self.assertTrue(bool((saved[0] == 3).all()))
        runtime.state[0][:, :, 3] = 1
        with self.assertRaisesRegex(ValueError, "Future KV overwritten"):
            capture_state(runtime, specs)

    def test_run_stage_keeps_teacher_and_greedy_requests_independent(self):
        runtime = CPURuntime(choices=[1, 2])
        result = run_stage(
            runtime,
            [dict(name="teacher", steps=[[0, 1], [1]])],
            [dict(name="chat", prompt=[0, 1], limit=5)],
            self.specs,
            lambda case, step: True,
            [2],
        )
        self.assertEqual(set(result.logits), {("teacher", 0), ("teacher", 1)})
        self.assertEqual(result.greedy["chat"]["ids"], [1, 2])
        self.assertEqual(runtime.resets, [0, 0])


if __name__ == "__main__":
    unittest.main()
