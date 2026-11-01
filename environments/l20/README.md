# L20 native stack

The profile describes the existing Linux x86-64 development stack:

| Component | Recorded value |
|---|---|
| Python | 3.12 |
| Torch runtime | 2.7.0+cu126 |
| Torch distribution metadata | 2.7.0 |
| Triton | 3.3.0 |
| CUDA used by Torch | 12.6 |
| Local nvcc used for the native build | 12.2.140 |
| GPU compute capability | 8.9 |

On the original development host, the native Python is
`/data/workspace/abo/.venv/bin/python`. Use that absolute path as `NATIVE_PY` in
the [shared workflow](../README.md), or supply an independently installed native
stack matching `profile.json`. The setup helper reads selected packages from
that environment and does not modify it.
Use the shared workflow's Bash entry points: `setup.sh` to create a model venv,
then `check.sh --python PATH` to inspect that model environment.

The profile's CUDA version is `torch.version.cuda`. The environment report
records the local CUDA compiler separately; its minor version can differ, as
in this existing stack. Retain the compiler and CMake build reports with each
deployment.

[`qwen3.lock`](qwen3.lock) preserves the complete original L20 Qwen3 freeze for
provenance. It includes model and native dependencies and is not a cross-target
installer. New model environments use the selected native stack plus their
model's `compiler/<model>/requirements.txt`.

The original full-model validation remains recorded under
[`docs/experiments/`](../../docs/experiments/). Re-run the shared validation matrix
for new packages. Qwen3.5 HF numerical acceptance remains open.

The new workflow was exercised on 2026-11-01: both model venvs were created
from the native stack and passed pip/profile/LibTorch-build checks. All seven
native targets and the attention library built. Qwen3's nine operator and nine
compiled-attention cases passed; Qwen3.5's GDN suite passed dispatch parity for
101 cases / 566 native calls / 1,698 tensors. Its 364 existing numerical failure
records matched the previous baseline exactly. Fifteen environment/setup tests
and the CPU cache regression also passed. Reports are in
`artifacts/l20/<model>/dual-target-20261101/`; this run did not repeat full-model
export or performance measurements.

After the GB10 fixture-layout correction, the standalone GDN suite was rerun
with AOTI input guards enabled on L20. All 101 cases / 566 native calls / 1,698
tensors still passed dispatch parity, with the same 364 numerical failure
records. That report is in the run's `cond-guarded/` directory.
