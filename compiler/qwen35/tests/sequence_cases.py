"""Boundary, partition and long-chain fixtures for the single-block experiment."""

import json

import torch


def fixtures(directory, config):
    generator = torch.Generator().manual_seed(35)
    hidden = torch.randn(1, 1024, config.hidden_size, generator=generator).bfloat16()
    inputs, cases = {}, []

    def add(name, parts, decode_steps=4, **tags):
        offset, steps = 0, []
        for index, length in enumerate(parts + [1] * decode_steps):
            key = f"{name}-{index}"
            inputs[key] = hidden[:, offset:offset + length].contiguous()
            offset += length
            steps.append(dict(kind="prefill" if index < len(parts) else "decode",
                              tokens=length, file=f"{key}.bin"))
        assert offset <= hidden.shape[1]
        cases.append(dict(name=name, steps=steps, prefix_steps=len(parts), **tags))

    for length in (1, 2, 3, 4, 5, 63, 64, 65, 127, 128, 129, 511, 512):
        add(f"length_{length}", [length])
    add("all_decode_65", [1] * 65, compare_to="length_65")
    add("shifted_129", [1, 64, 64], compare_to="length_129")
    add("mixed_129", [3, 61, 65], compare_to="length_129")
    add("long_decode", [512], decode_steps=512)
    add("repeat_65", [65], repeat_of="length_65")
    for name, value in inputs.items():
        (directory / f"{name}.bin").write_bytes(value.view(torch.uint8).numpy().tobytes())
    (directory / "cases.json").write_text(json.dumps(dict(cases=cases), indent=2) + "\n")
    return cases, inputs
