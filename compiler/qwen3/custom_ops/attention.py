"""Load Qwen3's native attention operator and register its shape contract."""

import os
from pathlib import Path
import subprocess


OPERATOR_CONTRACT = {"name": "xinfer::flash_attention", "abi_version": 1, "read_only": True}
_loaded = None


def native_attention_library(native):
    """Resolve the library used by a native worker under its sanitized environment."""
    env = {k: v for k, v in os.environ.items()
           if k not in ('VIRTUAL_ENV', 'PYTHONPATH', 'PYTHONHOME', 'LD_LIBRARY_PATH')}
    linked = subprocess.check_output(['ldd', str(Path(native).resolve())], env=env, text=True)
    for line in linked.splitlines():
        name, separator, target = line.strip().partition(' => ')
        if separator and name == 'libxinfer_attention_ops.so':
            library = Path(target.rsplit(' (', 1)[0])
            if not library.is_file():
                raise FileNotFoundError(f'Native attention dependency is missing: {line.strip()}')
            return library.resolve()
    return None


def load_attention_ops(library=None):
    import torch

    global _loaded
    library = Path(library or os.environ.get('XINFER_ATTENTION_OPS_LIBRARY') or
                   Path(__file__).resolve().parents[3] / 'build/libxinfer_attention_ops.so').resolve()
    if _loaded is not None:
        if _loaded != library:
            raise ValueError('Cannot load two different attention libraries in one Python process')
        return
    if not library.is_file():
        raise FileNotFoundError(f'Build the Qwen3 runtime first: {library}')
    torch.ops.load_library(str(library))

    @torch.library.register_fake('xinfer::flash_attention')
    def fake(q, k, v, kv_extent, scale):
        # The native kernel checks T <= L; output metadata depends only on Q.
        torch._check(q.dim() == 4)
        torch._check(q.shape[0] == 1)
        torch._check(q.shape[1] == 16)
        torch._check(q.shape[3] == 128)
        torch._check(k.shape == (1, 8, 1024, 128))
        torch._check(v.shape == k.shape)
        torch._check(q.dtype == k.dtype == v.dtype == torch.bfloat16)
        torch._check(q.device.type == 'cuda' and k.device == q.device and v.device == q.device)
        torch._check(kv_extent.device.type == 'cpu' and kv_extent.dtype == torch.uint8)
        torch._check(kv_extent.dim() == 1)
        return q.new_empty(q.shape)

    _loaded = library


def load_package_attention_ops(metadata, native=None):
    if metadata.get('attention_backend', 'efficient') != 'flash':
        raise ValueError('Legacy efficient packages are unsupported; re-export Qwen3 packages with the current exporter')
    if metadata.get('eager_numerics', False):
        raise ValueError('Legacy eager-numerics packages are unsupported; re-export Qwen3 packages with the current exporter')
    if 'custom_attention' not in metadata:
        raise ValueError('Flash packages require the custom-op contract; re-export Qwen3 packages with the current exporter')
    if metadata['custom_attention'] != OPERATOR_CONTRACT:
        raise ValueError('Unsupported custom attention contract')
    library = native_attention_library(native) if native is not None else None
    if native is not None and library is None:
        raise ValueError('Qwen3 runtime is missing its required native attention library; rebuild the runtime')
    load_attention_ops(library)
