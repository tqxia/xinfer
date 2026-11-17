# Kernel or optimization name

## Context

Identify the operation or fused region, workload/shapes/strides/dtypes, tested
toolchain, correctness reference and performance baseline.

## Design and implementation

Describe the measured bottleneck and supporting NCU/generated-code findings.
Explain the predicted mechanism and chosen design.
Include the essential code or a companion implementation, explaining the
relevant decomposition, dataflow, layout, scheduling or instruction choices.

## Significant revisions

Explain the changes tried, predicted versus observed effects and why each was
kept or rejected. Include useful failed attempts and changes of direction.

## Results and outcome

Give unprofiled baseline/candidate timings with units and measurement conditions.
Keep local kernel and native model results separate. Include correctness criteria,
results and remaining failures; prefill and memory effects when measured.
State what the evidence supports and what remains unresolved.

## Sources

Cite versioned source and related recipes. Keep the reasoning and essential
measurements in this case; do not link to local experiment records or temporary
reports.
