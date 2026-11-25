# PyTorch 2.13 patches

This directory owns compiler workarounds for PyTorch 2.13. Its name uses
underscores so the implementations are importable Python modules.

## AOTI constant-handle ownership

[`aoti_constant_handles.py`](aoti_constant_handles.py) fixes the user-managed
branch in `torch/csrc/inductor/aoti_runtime/utils.h`. That branch clones an
owning tensor handle without retaining it in an RAII owner. The patch checks
the clone operation and retains the cloned handle for release with its loader,
while continuing to share the caller's tensor storage.

The patch requires Torch `2.13.0`, source commit
`cf30153c4c131c8164ee7798e5022d810682e2cb`, and the exact header SHA-256 recorded
in the module. Unsupported versions, header contents or compiler include hooks
are rejected before compilation.

`compiler.common.compile` applies `owned_constant_handles()` around each AOTI
compile. It creates a temporary header overlay and restores the include hook
when compilation exits, including on failure. Installed Torch files are not
modified. The hook assumes sequential compilation. Existing bundles require
re-export to gain the fix; see the [shared-weight contract](../../../docs/package-metadata.md).

Run the shared-weight compilation and loading checks with the matching model
interpreter from the repository root:

```bash
CUDA_VISIBLE_DEVICES='' "$MODEL_PY" -m unittest compiler.common.tests.test_weights
```

## Strict export installed-global cleanup

[`export_cleanup.py`](export_cleanup.py) fixes the strict-export capture path
in `torch/_dynamo/convert_frame.py`. `DynamoOutput.graph_capture_output`
returns captured bytecode without registering `output_graph.cleanups` on
`CleanupManager.instance[self.bytecode]`, unlike `_compile`. Strict
`torch.export.export(..., strict=True)` therefore leaves `__compiled_fn_*`
references to all exported model parameters and buffers pinned in
`torch._dynamo.functional_export.__dict__` even after `torch._dynamo.reset()`
and garbage collection.

The patch requires Torch `2.13.0`, source commit
`cf30153c4c131c8164ee7798e5022d810682e2cb`, and the exact `convert_frame.py`
SHA-256 recorded in the module. Unsupported versions, source contents or
pre-patched hooks are rejected before export.

`compiler.qwen35.export` applies `cleanup_export_globals()` around each strict
`torch.export.export` call (covering both Qwen3.5 and Qwen3.8 export and
validation) and restores `DynamoOutput.graph_capture_output` on exit, including
on failure. Installed Torch files are not modified. The hook assumes sequential
exports.

Run the focused CPU tests with the matching model interpreter from the
repository root:

```bash
CUDA_VISIBLE_DEVICES='' "$MODEL_PY" -m unittest compiler.patches.torch_2_13.tests.test_export_cleanup
```
