"""Reusable Qwen3 handoff and split-KV correctness fixtures."""

from compiler.common.artifacts import require


def extra_cases(cases, seed):
    """Exercise native handoffs, split-KV transitions and poisoned future state."""
    require(bool(seed), "Hybrid validation needs a nonempty prompt seed")
    names = {case["name"] for case in cases}

    def prompt(length):
        return (seed * (length // len(seed) + 1))[:length]

    additions = [
        dict(name="hybrid_prompt_25", steps=[prompt(25), [0], [151645], [151935]]),
        dict(
            name="hybrid_back_to_aoti",
            steps=[prompt(2), [13], prompt(13), [0], prompt(2), [151643]],
        ),
    ]
    clean = additions[1]
    additions.append(
        dict(
            name=clean["name"] + "_poison",
            steps=clean["steps"],
            same_as=clean["name"],
            poison_kv=3,
        )
    )
    for boundary in (128, 256):
        clean = dict(
            name=f"hybrid_split_{boundary}",
            steps=[prompt(boundary - 1), [13], [0], prompt(2), [151643]],
        )
        additions.extend(
            (
                clean,
                dict(
                    name=clean["name"] + "_poison",
                    steps=clean["steps"],
                    same_as=clean["name"],
                    poison_kv=3,
                ),
            )
        )
    for order, case in enumerate(additions, start=100):
        require(case["name"] not in names, "Duplicate hybrid fixture")
        case.update(order=order, dump_cache=True)
        names.add(case["name"])
    return [*cases, *additions]
