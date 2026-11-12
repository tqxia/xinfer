# From profile observations to techniques

Use this after establishing the baseline profile. Use measured costs to guide
fusion and kernel experiments. Load only the relevant note; these are starting
questions, not an exhaustive catalog or a rule that a metric mandates a technique.

| Observed behavior | Starting direction | Read |
| --- | --- | --- |
| Repeated small launches or intermediate copies/writes in one region | Can fusion or projection packing reduce launches, traffic or repeated loads? | [Fusion](techniques/common.md#fusion) |
| Matrix/reduction calls dominate the target phase | Would tuning, another compiler/library implementation or a rewritten kernel improve the complete region? | [Kernel optimization](techniques/common.md#kernel-optimization) |
| Inefficient memory transactions, spills or insufficient parallel work | Is the layout or schedule unsuitable for these inputs? | [Layout and scheduling](techniques/common.md#layout-and-scheduling) |
| A proposed mechanism needs target-specific instructions or resources | Does the exact GPU and installed toolchain support it, and can this workload amortize its cost? | [Hardware-specific mechanisms](techniques/hardware.md) |

Check whether the compiler already performs the proposed transformation.
Graph passes, inference adapters and custom operators can implement any of these
goals. An unexplained observation calls for more investigation, not a forced match
to this table. Runtime-only findings remain opportunities for the future skill.

Common mechanisms still need target measurements; hardware-specific recipes
also need explicit feature/version applicability. Grow these notes from actual
experiments, retaining useful limitations and failed approaches with their
context. Keep new evidence and predicted-versus-observed feedback in the
experiment notes first; add a shared lesson only when it is reusable.
