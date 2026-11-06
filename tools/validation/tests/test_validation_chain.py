"""CPU checks for validation-stage coverage, policies and request orchestration."""

import unittest

import torch

from tools.validation.common.chain import Policy, compare_traces, new_report
from tools.validation.common.runner import Trace, greedy, run_stage, teacher_trace


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

    def test_fp32_diagnostics_do_not_inherit_required_state_or_token_gates(self):
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
        fp32 = self.compare(actual, reference, policy=Policy(tokens_required=True), required=False)
        self.assertEqual(fp32["failures"], [])
        self.assertEqual(fp32["greedy_failures"], [])


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
