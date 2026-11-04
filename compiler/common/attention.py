"""Load the shared native prefix-Flash operator and register its shape contract."""

import os
from pathlib import Path
import subprocess

from compiler.common.artifacts import default_build_dir, require


ATTENTION_OPS_ABI_VERSION = 1
_loaded = None


def cache_extent(valid_length):
    """Only this CPU tensor's shape is read; +1 avoids AOTI's size-one guards."""
    import torch

    return torch.empty(valid_length + 1, dtype=torch.uint8)


def native_attention_library(native):
    """Resolve the library used by a native worker under its sanitized environment."""
    env = {
        k: v
        for k, v in os.environ.items()
        if k not in ("VIRTUAL_ENV", "PYTHONPATH", "PYTHONHOME", "LD_LIBRARY_PATH")
    }
    linked = subprocess.check_output(["ldd", str(Path(native).resolve())], env=env, text=True)
    for line in linked.splitlines():
        name, separator, target = line.strip().partition(" => ")
        if separator and name == "libxinfer_attention_ops.so":
            library = Path(target.rsplit(" (", 1)[0])
            if not library.is_file():
                raise FileNotFoundError(f"Native attention dependency is missing: {line.strip()}")
            return library.resolve()
    return None


def load_attention_ops(library=None, *, model=None):
    global _loaded
    selected = library or os.environ.get("XINFER_ATTENTION_OPS_LIBRARY")
    if not selected and _loaded is not None:
        return
    if not selected:
        if model is None:
            raise ValueError("Select a model to locate its default native attention library")
        selected = default_build_dir(model) / "libxinfer_attention_ops.so"
    library = Path(selected).resolve()
    if _loaded is not None:
        if _loaded != library:
            raise ValueError("Cannot load two different attention libraries in one Python process")
        return
    if not library.is_file():
        raise FileNotFoundError(f"Build the native attention library first: {library}")
    import torch

    torch.ops.load_library(str(library))
    try:
        version = torch.ops.xinfer.attention_ops_abi_version()
    except (AttributeError, RuntimeError) as error:
        raise ValueError("Native attention ABI query is missing; rebuild the runtime") from error
    require(
        type(version) is int and version == ATTENTION_OPS_ABI_VERSION,
        "Unsupported native attention ABI",
    )

    @torch.library.register_fake("xinfer::flash_attention")
    def fake(q, k, v, kv_extent, scale):
        # Valid-prefix length is checked natively; output metadata depends only on Q.
        torch._check(q.dim() == 4 and k.dim() == 4 and v.dim() == 4)
        torch._check(q.shape[0] == 1)
        torch._check(q.shape[1] > 0)
        torch._check(k.shape[1] > 0)
        torch._check(q.shape[1] % k.shape[1] == 0)
        torch._check(q.shape[3] >= 8)
        torch._check(q.shape[3] <= 256)
        torch._check(q.shape[3] % 8 == 0)
        torch._check(q.shape[2] >= 1)
        torch._check(q.shape[2] <= 512)
        torch._check(k.shape == (1, k.shape[1], 1024, q.shape[3]))
        torch._check(v.shape == k.shape)
        torch._check(q.dtype == k.dtype == v.dtype == torch.bfloat16)
        torch._check(q.device.type == "cuda" and k.device == q.device and v.device == q.device)
        torch._check(kv_extent.device.type == "cpu" and kv_extent.dtype == torch.uint8)
        torch._check(kv_extent.dim() == 1)
        return q.new_empty(q.shape)

    _loaded = library
