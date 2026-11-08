"""Select the bundled attention operator for the eager validation stage."""

from pathlib import Path
import tempfile
import zipfile

from compiler.common import attention
from compiler.common.artifacts import require


_packaged_directory = None


def _packaged_attention(package):
    with zipfile.ZipFile(package) as archive:
        records = [
            name
            for name in archive.namelist()
            if "/data/aotinductor/" in name and name.endswith("/" + attention.ATTENTION_LIBRARY)
        ]
        require(
            len(records) == 1, "Package is missing its attention library; re-export both entries"
        )
        payload = archive.read(records[0])
        require(bool(payload), "Packaged attention library is empty")
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
    payload = _packaged_attention(Path(packages) / "prefill.pt2")
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
