# Hardware-specific mechanisms

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
