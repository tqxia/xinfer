"""Register the native, opaque read-only attention op and its shape contract."""

from pathlib import Path

import torch


OPERATOR_CONTRACT = {"name": "xinfer::flash_attention", "abi_version": 1, "read_only": True}
_loaded = False


def load_attention_ops():
    global _loaded
    if _loaded:
        return
    library = Path(__file__).resolve().parents[2] / "build/libxinfer_attention_ops.so"
    if not library.is_file():
        raise FileNotFoundError(f"Build scripts/build-gb10.sh first: {library}")
    torch.ops.load_library(str(library))

    @torch.library.register_fake("xinfer::flash_attention")
    def fake(q, k, v, kv_extent, scale):
        # Do not run Flash or form dynamic KV views here. The runtime kernel
        # checks T <= L; output metadata depends only on Q, not on L.
        torch._check(q.dim() == 4)
        torch._check(q.shape[0] == 1)
        torch._check(q.shape[1] == 16)
        torch._check(q.shape[3] == 128)
        torch._check(k.shape == (1, 8, 1024, 128))
        torch._check(v.shape == k.shape)
        torch._check(q.dtype == k.dtype == v.dtype == torch.bfloat16)
        torch._check(q.device.type == "cuda" and k.device == q.device and v.device == q.device)
        torch._check(kv_extent.device.type == "cpu" and kv_extent.dtype == torch.uint8)
        torch._check(kv_extent.dim() == 1)
        return q.new_empty(q.shape)

    _loaded = True


def load_package_attention_ops(metadata):
    # flash-custom is accepted only as a legacy package label, not an exporter.
    if metadata.get("attention_backend") in ("flash", "flash-custom"):
        if "custom_attention" not in metadata:
            raise ValueError("Flash packages require the custom-op graph rewrite; re-export with --attention-backend flash")
        if metadata.get("custom_attention") != OPERATOR_CONTRACT:
            raise ValueError("Unsupported custom attention contract")
        load_attention_ops()
