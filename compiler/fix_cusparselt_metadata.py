"""Repair the incorrect inner platform tag in NVIDIA's ARM64 0.8.1 wheel.

The published filename is manylinux2014_aarch64, but WHEEL says
manylinux2014_sbsa. pip 26 rejects that inner tag during `pip check`.
Only installation metadata is changed; the shared library remains untouched.
"""

import base64
import csv
import hashlib
from importlib.metadata import distribution
import io
import platform
import sys


def main():
    if platform.system() != "Linux" or platform.machine() != "aarch64":
        return
    if sys.prefix == sys.base_prefix:
        raise RuntimeError("Run inside the project virtual environment")
    dist = distribution("nvidia-cusparselt-cu13")
    if dist.version != "0.8.1":
        return
    wheel_entry = next(p for p in dist.files if str(p).endswith(".dist-info/WHEEL"))
    wheel = dist.locate_file(wheel_entry)
    text = wheel.read_text()
    bad = "Tag: py3-none-manylinux2014_sbsa"
    if bad not in text:
        return
    library_entry = next(p for p in dist.files if str(p).endswith("/libcusparseLt.so.0"))
    with dist.locate_file(library_entry).open("rb") as stream:
        header = stream.read(20)
    if header[:6] != b"\x7fELF\x02\x01" or int.from_bytes(header[18:20], "little") != 183:
        raise RuntimeError("Expected a 64-bit little-endian AArch64 ELF library")
    data = text.replace(bad, "Tag: py3-none-manylinux2014_aarch64").encode()
    record = wheel.with_name("RECORD")
    rows = list(csv.reader(io.StringIO(record.read_text())))
    row = next(row for row in rows if row[0] == str(wheel_entry))
    digest = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=").decode()
    row[1:] = ["sha256=" + digest, str(len(data))]
    wheel.write_bytes(data)
    with record.open("w", newline="") as stream:
        csv.writer(stream).writerows(rows)
    print("Corrected cuSPARSELt 0.8.1 WHEEL tag to match its ARM64 binary and wheel filename")


if __name__ == "__main__":
    main()
