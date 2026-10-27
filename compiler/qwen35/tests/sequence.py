"""Run variable-length block inputs with the existing T=64 and T=1 packages."""

import torch

from compiler.qwen35.config import PREFILL


MAX_TOKENS = 512


def chunks(hidden):
    if hidden.ndim != 3 or hidden.shape[0] != 1 or not 1 <= hidden.shape[1] <= MAX_TOKENS:
        raise ValueError("Expected hidden [1, T, H] with 1 <= T <= 512")
    offset = 0
    while offset < hidden.shape[1]:
        length = PREFILL if hidden.shape[1] - offset >= PREFILL else 1
        yield hidden[:, offset:offset + length].contiguous()
        offset += length


def run_sequence(functions, hidden, conv, recurrent):
    outputs = []
    for piece in chunks(hidden):
        kind = "prefill" if piece.shape[1] == PREFILL else "decode"
        output, conv, recurrent = functions[kind](piece, conv, recurrent)
        outputs.append(output)
    return (torch.cat(outputs, dim=1) if len(outputs) > 1 else outputs[0], conv, recurrent)
