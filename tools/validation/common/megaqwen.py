"""Select the bundled MegaQwen operator for the eager validation stage."""

from pathlib import Path
import tempfile
import zipfile

from compiler.common.artifacts import require
from compiler.qwen3 import megaqwen


_packaged_directory = None


def load_packaged_megaqwen_ops(packages):
    """Load decode's implementation and FakeTensor contract for eager validation.

    Ordinary Python/C++ AOTI loading needs no helper: the wrapper loads its
    native dependency. Reuse an already selected eager implementation and keep
    an extracted library alive for the process.
    """
    global _packaged_directory
    try:
        return megaqwen.megaqwen_library()
    except ValueError:
        pass
    with zipfile.ZipFile(Path(packages) / "decode.pt2") as archive:
        records = [
            name
            for name in archive.namelist()
            if "/data/aotinductor/" in name and name.endswith("/" + megaqwen.MEGAQWEN_LIBRARY)
        ]
        require(
            len(records) == 1, "Package is missing its MegaQwen library; re-export both entries"
        )
        payload = archive.read(records[0])
        require(bool(payload), "Packaged MegaQwen library is empty")
    temporary = tempfile.TemporaryDirectory(prefix="xinfer-megaqwen-")
    library = Path(temporary.name) / megaqwen.MEGAQWEN_LIBRARY
    library.write_bytes(payload)
    try:
        megaqwen.load_megaqwen_ops(library)
    except Exception:
        temporary.cleanup()
        raise
    _packaged_directory = temporary
    return library
