# Fusion

Look for related launches and global-memory intermediates in the measured
region. Consider compiler fusion/combo options or a custom fused kernel;
independent projections sharing an input may suit packing. Check existing
compiler/pass coverage first; source-op count is not launch count.

Predict savings in launches, intermediates or repeated loads. The required
arithmetic may remain; describe any redundant computation removed separately.
Check whether added copies, workspace, register pressure or changed rounding
boundaries offset the saving. Measure the whole replacement, including its
preparation, then use the remaining profile costs to guide another iteration.
