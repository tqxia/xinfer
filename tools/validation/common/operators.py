"""Select bundled native operators for the eager validation stage."""

from pathlib import Path
import tempfile
import zipfile

from compiler.common.kernels.flash_attention import attention
from compiler.common.artifacts import require


_packaged_directory = None
_packaged_deltanet_directory = None


def _packaged_library(package, filename, label):
    with zipfile.ZipFile(package) as archive:
        records = [
            name
            for name in archive.namelist()
            if "/data/aotinductor/" in name and name.endswith("/" + filename)
        ]
        require(
            len(records) == 1, f"Package is missing its {label} library; re-export both entries"
        )
        payload = archive.read(records[0])
        require(bool(payload), f"Packaged {label} library is empty")
        return payload


def load_packaged_attention_ops(packages):
    """Select prefill's packaged operator for eager validation and shape registration.

    Reuse an already registered eager operator. Ordinary Python/C++ AOTI loading
    needs no helper: the wrapper loads its native dependency. Keep this eager
    handle for the process.
    """
    global _packaged_directory
    try:
        return attention.attention_library()
    except ValueError:
        pass
    payload = _packaged_library(
        Path(packages) / "prefill.pt2", attention.ATTENTION_LIBRARY, "attention"
    )
    temporary = tempfile.TemporaryDirectory(prefix="xinfer-attention-")
    library = Path(temporary.name) / attention.ATTENTION_LIBRARY
    library.write_bytes(payload)
    try:
        attention.load_attention_ops(library)
    except Exception:
        temporary.cleanup()
        raise
    _packaged_directory = temporary
    return library


def load_packaged_deltanet_ops(packages):
    """Select prefill's packaged DeltaNet operator for eager Qwen3.5 validation."""
    global _packaged_deltanet_directory
    from compiler.common.kernels.gated_delta_net import deltanet

    try:
        return deltanet.deltanet_library()
    except ValueError:
        pass
    payload = _packaged_library(
        Path(packages) / "prefill.pt2", deltanet.DELTANET_LIBRARY, "DeltaNet"
    )
    temporary = tempfile.TemporaryDirectory(prefix="xinfer-deltanet-")
    library = Path(temporary.name) / deltanet.DELTANET_LIBRARY
    library.write_bytes(payload)
    try:
        deltanet.load_deltanet_ops(library)
    except Exception:
        temporary.cleanup()
        raise
    _packaged_deltanet_directory = temporary
    return library
