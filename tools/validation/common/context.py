"""Matched long-context fixtures and bounded state checkpoint capture."""

from compiler.common.artifacts import require
from tools.common import prompt_chunks


def state_shape(spec, position, compact):
    shape = list(spec["shape"])
    if compact and spec["update"] == "inplace":
        shape[2] = position
    return shape


def long_context_cases(seed, capacity, max_prefill):
    """Teacher inputs expose every graph call; only selected long states are saved."""
    if capacity <= 1024:
        return []
    require(seed, "Missing long-context seed")
    stream = (seed * (capacity // len(seed) + 1))[:capacity]
    cases = []
    for length in (1, 511, 512, 513, 8191, 8192):
        if length + 2 > capacity:
            continue
        steps = prompt_chunks(stream[:length], max_prefill) + [
            [stream[length]],
            [stream[length + 1]],
        ]
        cases.append(dict(name=f"long_prompt_{length}", steps=steps, long_context=True))
    sizes = (511, 512, 1, 513)
    if sum(sizes) <= capacity:
        offset, steps = 0, []
        for length in sizes:
            steps.extend(prompt_chunks(stream[offset : offset + length], max_prefill))
            offset += length
        cases.append(dict(name="long_nonzero_offsets", steps=steps, long_context=True))
    # A final one-token call exercises decode at the last valid cache position.
    steps = prompt_chunks(stream[:-1], max_prefill) + [stream[-1:]]
    cases.append(dict(name="long_context_end", steps=steps, check_overflow=True, long_context=True))
    clean = next((case for case in cases if case["name"] == "long_prompt_513"), None)
    if clean is not None:
        cases.extend(
            (
                dict(clean, name="long_prompt_513_poison", same_as=clean["name"], poison_kv=3),
                dict(clean, name="long_prompt_513_repeat", same_as=clean["name"]),
            )
        )
    return cases


def capture_long_state(case, step):
    """Bound long traces to the completed prompt/request, keeping every logit."""
    return step in {max(0, len(case["steps"]) - 3), len(case["steps"]) - 1}
