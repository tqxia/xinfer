# PyTorch 2.13 patches

Version- and SHA-256-guarded compiler workarounds for Torch `2.13.0` (source
commit `cf30153c4c131c8164ee7798e5022d810682e2cb`). Each patch checks exact file
digests before applying, restores hooks on exit (including failures), assumes
sequential compilation/export, and never modifies installed Torch files.

## AOTI constant-handle ownership

[`aoti_constant_handles.py`](aoti_constant_handles.py) patches the user-managed
branch in `torch/csrc/inductor/aoti_runtime/utils.h` so cloned constant handles
are retained in an RAII owner and released with their loader while continuing to
borrow the caller's shared tensor storage. `compiler.common.compile` applies
`owned_constant_handles()` via a temporary header overlay around each AOTI
compile; see [package metadata](../../../docs/package-metadata.md).

```bash
CUDA_VISIBLE_DEVICES='' "$PY" -m unittest compiler.common.tests.test_weights
```

## Strict export installed-global cleanup

[`export_cleanup.py`](export_cleanup.py) patches
`DynamoOutput.graph_capture_output` in `torch/_dynamo/convert_frame.py` to
register `output_graph.cleanups` on `CleanupManager`, preventing strict
`torch.export.export(..., strict=True)` from pinning `__compiled_fn_*`
references to exported parameters in `torch._dynamo.functional_export.__dict__`.
`compiler.qwen35.export` applies `cleanup_export_globals()` around strict export
calls for Qwen3.5 and Qwen3.8.

```bash
CUDA_VISIBLE_DEVICES='' "$PY" -m unittest compiler.patches.torch_2_13.tests.test_export_cleanup
```
