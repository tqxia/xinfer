# Hardware-specific guidance

Consult the relevant target notes before autotuning and timing. A quiet GPU is
not necessarily healthy under load: sample clocks, utilization, power and clock
event reasons during a short representative workload. Interpret them against the
selected device's healthy behavior, not a universal frequency threshold.

## Target notes

### GB10 (SM 12.1)

One tested system stayed at 611 MHz under load despite passing idle checks.
After shutdown and PSU power disconnection, its busy samples returned to roughly
2.4–2.5 GHz. This is an observed failure/recovery on that system, not a normal
GB10 clock target or a proven root cause shared by all GB10 devices.

Check load telemetry before export/autotuning as well as benchmarking. After
recovering from abnormal clocks, reassess artifacts whose tuning choices were
selected under the fault; the recorded experiment re-exported its baseline.
Evidence is in `docs/experiments/gb10-qwen35-perf-plugin-2026-11-10.md` in the
selected xinfer checkout.

### L20 (SM 8.9)

One tested L20 held 2,520 MHz during native timing. Its sustained matrix-multiply
probe ran at 2,460–2,505 MHz with software power-capping event `0x4`. Interpret
clock events against the workload, power and temperature; do not require every
event flag to be zero or import GB10's frequency observations as a threshold.
The supporting experiment report remains local for now. These observations do
not define universal L20 clocks.

## Feature-dependent mechanisms

Read this when a hypothesis depends on features such as TMA, specialized matrix
instructions, tensor memory or architecture-specific asynchronous pipelines.
General tiling, warp-role scheduling and fusion ideas remain common mechanisms;
their instruction-level implementation can be hardware-specific.

Before adopting a recipe, establish:

- The exact GPU/compute capability and installed CUDA, Triton and PyTorch stack.
- The source example's supported architecture, API/version and resource needs.
- The profile observation it could address and whether the workload has enough
  work or reuse to offset its setup, synchronization and workspace costs.

Architecture names are insufficient: GB10 and B200 are listed with different
compute capabilities in [NVIDIA's device table](https://developer.nvidia.com/cuda/gpus).
Verify applicability to the actual target rather than inferring it from a
marketing family. Consult matching source/documentation before implementing.
The [Triton persistent-matmul examples](https://triton-lang.org/main/getting-started/tutorials/09-persistent-matmul.html)
illustrate variants with different feature/resource requirements.

There are no preselected device recipes here yet. After real experiments, add
only useful target-specific findings with the GPU/SM, toolchain, workload,
observed effect and applicability limits; a result on one target is not evidence
of a gain on another. Keep the existing package and runtime contract intact.
