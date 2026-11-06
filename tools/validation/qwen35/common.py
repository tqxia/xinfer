"""Full-model request fixtures reused by runtime contract checks."""

from tools.validation.common.prompts import PARAGRAPH


def fixtures(tokenizer):
    paragraph = tokenizer.encode(
        PARAGRAPH,
        add_special_tokens=False,
    )
    stream = (paragraph * (1024 // len(paragraph) + 1))[:1024]

    def case(name, prefix, decode):
        sizes = [prefix] + [1] * decode
        offset, steps = 0, []
        for length in sizes:
            steps.append(stream[offset : offset + length])
            offset += length
        return dict(name=name, prefix=prefix, steps=steps)

    cases = [
        case("decode_zero", 1, 3),
        case("prefill_decode", 64, 8),
        case("remainder_65", 65, 4),
        case("two_prefills", 128, 4),
        case("long_prefix", 512, 8),
        case("repeat", 64, 8),
    ]
    cases += [
        case(f"dynamic_{length}", length, 2)
        for length in (2, 22, 31, 32, 33, 63, 96, 97, 98, 127, 129, 511)
    ]
    sizes = (96, 98, 97, 99, 1)
    offset, steps = 0, []
    for length in sizes:
        steps.append(stream[offset : offset + length])
        offset += length
    mixed = dict(name="mixed_dispatch", steps=steps)
    cases.extend((mixed, dict(mixed, name="mixed_dispatch_repeat")))
    return cases
