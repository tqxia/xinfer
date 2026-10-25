"""Optional native FlashInfer decode registration; no FlashInfer Python runtime."""

import torch

from .attention import load_attention_ops

SOURCE = {"version": "0.7.2.post1", "revision": "beabf5004585b03da60e927bbb803acbaa62fe36",
          "implementation": "single_decode_cuda_core"}
OPERATOR_CONTRACT = {"name": "xinfer::flashinfer_decode", "abi_version": 1, "read_only": True}
_loaded = False


def load_flashinfer_ops():
    global _loaded
    if _loaded:
        return
    load_attention_ops()
    if (not hasattr(torch.ops.xinfer, "flashinfer_revision") or
            torch.ops.xinfer.flashinfer_revision() != SOURCE["revision"]):
        raise ValueError("Build with -DQWEN_ENABLE_FLASHINFER=ON and the pinned FlashInfer source first")

    @torch.library.register_fake("xinfer::flashinfer_decode")
    def fake(q, k, v, kv_extent, scale):
        torch._check(q.shape == (1, 16, 1, 128))
        torch._check(k.shape == (1, 8, 1024, 128) and v.shape == k.shape)
        torch._check(q.dtype == k.dtype == v.dtype == torch.bfloat16)
        torch._check(q.device.type == "cuda" and k.device == q.device and v.device == q.device)
        torch._check(kv_extent.device.type == "cpu" and kv_extent.dtype == torch.uint8)
        torch._check(kv_extent.dim() == 1)
        return q.new_empty(q.shape)

    _loaded = True


def load_package(metadata):
    from .attention import OPERATOR_CONTRACT as FLASH_CONTRACT
    kind = metadata.get("entry_point")
    expected = OPERATOR_CONTRACT if kind == "decode" else FLASH_CONTRACT
    if kind not in ("prefill", "decode") or metadata.get("custom_attention") != expected:
        raise ValueError("Unsupported custom attention contract")
    if metadata.get("flashinfer_source") != SOURCE:
        raise ValueError("Unsupported FlashInfer source contract")
    load_flashinfer_ops()
