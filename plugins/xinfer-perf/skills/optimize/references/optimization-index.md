# From profile observations to techniques

Use this after establishing the baseline profile. Load only the note relevant
to the observation; these are starting questions, not an exhaustive catalog or
a rule that a particular metric mandates a technique.

| Observed behavior | Starting direction | Read |
| --- | --- | --- |
| Repeated small launches or intermediate copies/writes in one region | Can compiler fusion, graph simplification or packing eliminate actual compiled work? | [Combine work](techniques/common.md#remove-or-combine-work) |
| Matrix/reduction calls dominate the target phase | Would another supported compiler configuration or library/generated implementation improve the complete region? | [Execution choices](techniques/common.md#choose-compiler-or-library-execution) |
| Inefficient memory transactions, spills or insufficient parallel work | Is the layout or schedule unsuitable for these inputs? | [Layout and scheduling](techniques/common.md#adjust-layout-and-scheduling) |
| A proposed mechanism needs target-specific instructions or resources | Does the exact GPU and installed toolchain support it, and can this workload amortize its cost? | [Hardware-specific mechanisms](techniques/hardware.md) |

Check whether the compiler already performs the proposed transformation.
An unexplained observation calls for more investigation, not a forced match
to this table. Runtime-only findings remain opportunities for the future skill.

Common mechanisms still need target measurements; hardware-specific recipes
also need explicit feature/version applicability. Grow these notes from actual
experiments, retaining useful limitations and failed approaches with their
context. Keep new evidence and predicted-versus-observed feedback in the
experiment notes first; add a shared lesson only when it is reusable.
