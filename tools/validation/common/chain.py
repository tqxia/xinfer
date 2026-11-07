"""One comparison policy and report format for the HF/eager/AOTI chain."""

from dataclasses import dataclass

from compiler.common.artifacts import require
from compiler.common.metadata import state_name
from tools.validation.common.context import state_shape
from tools.validation.common.validation import (
    DEFAULT_RELATIVE_LIMIT,
    tensor_metrics,
    token_differences,
    within_limits,
)


@dataclass
class Policy:
    state_required: bool = False
    tokens_required: bool = False
    limit: float = DEFAULT_RELATIVE_LIMIT
    state_rows: object = None
    token_keys: object = None

    def rows(self, tensor, spec):
        if self.state_rows is not None:
            return self.state_rows(tensor, spec)
        width = tensor.shape[-1] if spec["kind"] == "conv" else tensor.shape[-2] * tensor.shape[-1]
        return tensor.float().reshape(-1, width)


def new_report(model_type, metadata=None):
    report = dict(
        schema_version=1,
        model_type=model_type,
        stages=[],
        comparisons=[],
        failures=[],
        diagnostics=[],
        top1_differences=[],
        greedy_failures=[],
        greedy_differences=[],
        native_bitwise=False,
        passed=False,
    )
    if metadata:
        report.update(
            checkpoint=metadata["decode"]["checkpoint"],
            package_sha256={kind: entry["package_sha256"] for kind, entry in metadata.items()},
            shared_weights=metadata["decode"]["shared_weights"],
        )
    return report


def _keys(actual, expected, scope, label):
    if scope is None:
        require(actual.keys() == expected.keys(), f"Missing or extra {label} steps")
        return set(actual)
    require(set(scope) <= actual.keys() and set(scope) <= expected.keys(), f"Missing {label} steps")
    return set(scope)


def _record(report, row, failed):
    report["comparisons"].append(row)
    if failed:
        report["failures" if row["required"] else "diagnostics"].append(row)


def compare_traces(
    report,
    actual_name,
    expected_name,
    actual,
    expected,
    specs,
    policy,
    *,
    exact=False,
    required=True,
    state_keys=None,
    logit_keys=None,
    check_tokens=True,
    greedy_names=None,
    greedy_state_names=(),
):
    label = f"{actual_name}_vs_{expected_name}"
    logits = _keys(actual.logits, expected.logits, logit_keys, "logit")
    states = _keys(actual.states, expected.states, state_keys, "state")
    generations = _keys(actual.greedy, expected.greedy, greedy_names, "generation")
    require(logits or states or generations, "Empty stage comparison")

    def compare_tensor(a, b, case, step, tensor, *, state_spec=None):
        rows = None
        if state_spec is not None:
            rows = policy.rows(a, state_spec), policy.rows(b, state_spec)
        else:
            rows = a.float(), b.float()
        error = tensor_metrics(a, b, rows=rows)
        mandatory = required and (exact or state_spec is None or policy.state_required)
        row = dict(
            comparison=label,
            case=case,
            step=step,
            tensor=tensor,
            required=mandatory,
            exact=exact,
            **error,
        )
        _record(
            report,
            row,
            not error["bitwise_equal"] if exact else not within_limits(error, policy.limit),
        )
        return error

    for case, step in sorted(logits):
        a, b = actual.logits[case, step], expected.logits[case, step]
        compare_tensor(a, b, case, step, "logits")
        if exact and actual.raw_logits:
            require(
                (case, step) in actual.raw_logits and (case, step) in expected.raw_logits,
                "Missing raw logit parity checkpoint",
            )
            raw_error = tensor_metrics(
                actual.raw_logits[case, step],
                expected.raw_logits[case, step],
                rows=(
                    actual.raw_logits[case, step].float(),
                    expected.raw_logits[case, step].float(),
                ),
            )
            _record(
                report,
                dict(
                    comparison=label,
                    case=case,
                    step=step,
                    tensor="raw_logits",
                    required=required,
                    exact=True,
                    **raw_error,
                ),
                not raw_error["bitwise_equal"],
            )
        if check_tokens:
            differences = token_differences(a, b)
            if differences:
                mandatory = required and (
                    exact
                    or (
                        policy.tokens_required
                        and (policy.token_keys is None or (case, step) in policy.token_keys)
                    )
                )
                row = dict(
                    comparison=label,
                    case=case,
                    step=step,
                    tensor="tokens",
                    required=mandatory,
                    differences=differences,
                )
                report["top1_differences"].append(row)
                if mandatory:
                    report["failures"].append(row)

    for case, step in sorted(states):
        left, right = actual.states[case, step], expected.states[case, step]
        require(len(left) == len(right) == len(specs), "Incorrect state tensor count")
        if actual.compact_states or expected.compact_states:
            require(
                actual.compact_states == expected.compact_states, "State capture policies differ"
            )
            require(
                actual.positions[case, step] == expected.positions[case, step],
                "State positions differ",
            )
        position = actual.positions.get((case, step), 0)
        for a, b, spec in zip(left, right, specs):
            require(
                list(a.shape)
                == list(b.shape)
                == state_shape(spec, position, actual.compact_states),
                "State tensor shapes differ",
            )
            compare_tensor(a, b, case, step, state_name(spec), state_spec=spec)

    for name in sorted(generations):
        a, b = actual.greedy[name], expected.greedy[name]
        require(
            len(a["rows"]) == len(a["ids"]) and len(b["rows"]) == len(b["ids"]),
            "Generation rows and tokens disagree",
        )
        if a["ids"] != b["ids"]:
            mandatory = required and (exact or policy.tokens_required)
            first = next(
                (i for i, (x, y) in enumerate(zip(a["ids"], b["ids"])) if x != y),
                min(len(a["ids"]), len(b["ids"])),
            )
            row = dict(
                comparison=label,
                case=name,
                tensor="greedy_tokens",
                required=mandatory,
                first_difference=first,
                actual_ids=a["ids"],
                expected_ids=b["ids"],
            )
            report["greedy_differences"].append(row)
            if mandatory:
                report["greedy_failures"].append(row)
                report["failures"].append(row)
        if exact:
            require(len(a["rows"]) == len(b["rows"]), "Native generation length differs")
            require(a["position"] == b["position"], "Native generation position differs")
            for step, (x, y) in enumerate(zip(a["rows"], b["rows"])):
                compare_tensor(x, y, name, step, "logits")
        if name in greedy_state_names:
            require(
                len(a["state"]) == len(b["state"]) == len(specs), "Incorrect greedy state count"
            )
            for x, y, spec in zip(a["state"], b["state"], specs):
                compare_tensor(x, y, name, "final", state_name(spec), state_spec=spec)
