"""Build-only ownership fix for the pinned Torch 2.13 AOTI runtime header."""

from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import tempfile

import torch

from compiler.common.artifacts import require


TORCH_GIT = "cf30153c4c131c8164ee7798e5022d810682e2cb"
HEADER_SHA256 = "09fa306d3134a95d5d0fdf27d4b7d2ee14d7ff9b9ea98996dd79b7e78e1be961"
HEADER = Path("torch/csrc/inductor/aoti_runtime/utils.h")
ORIGINAL = """    if (user_managed) {
      aoti_torch_new_tensor_handle(handle, &handle_);
    } else {"""
FIXED = """    if (user_managed) {
      AOTI_TORCH_ERROR_CODE_CHECK(
          aoti_torch_new_tensor_handle(handle, &handle_));
      raii_handle_ = RAIIAtenTensorHandle(handle_);
    } else {"""


@contextmanager
def owned_constant_handles():
    """Own cloned handles while continuing to share the caller's tensor storage.

    Torch's user-managed branch creates a new owning at::Tensor handle but does
    not delete it. Compile packages against a guarded header overlay; never
    alter installed headers or the original caller handle. Existing packages
    must be recompiled. This scoped compiler hook assumes sequential compiles.
    """
    from torch._inductor.cpp_builder import BuildOptionsBase, CppTorchDeviceOptions

    require(
        str(torch.__version__).split("+", 1)[0] == "2.13.0"
        and torch.version.git_version == TORCH_GIT,
        "AOTI handle ownership workaround requires the pinned Torch 2.13 build",
    )
    source_path = Path(torch.__file__).resolve().parent / "include" / HEADER
    source = source_path.read_text()
    require(
        hashlib.sha256(source.encode()).hexdigest() == HEADER_SHA256
        and source.count(ORIGINAL) == 1,
        "Unsupported Torch AOTI ownership header",
    )
    original_getter = CppTorchDeviceOptions.get_include_dirs
    require(
        original_getter is BuildOptionsBase.get_include_dirs
        and "get_include_dirs" not in CppTorchDeviceOptions.__dict__,
        "Unsupported Torch AOTI include configuration",
    )
    fixed = source.replace(ORIGINAL, FIXED)
    with tempfile.TemporaryDirectory(prefix="xinfer-aoti-headers-") as temporary:
        overlay = Path(temporary)
        target = overlay / HEADER
        target.parent.mkdir(parents=True)
        target.write_text(fixed)

        def include_dirs(options):
            existing = original_getter(options)
            return [str(overlay), *existing] if options._aot_mode else existing

        # Include order participates in AOTI's C++ cache key. Its PCH builder
        # also hashes preprocessed header contents, selecting a corrected PCH.
        CppTorchDeviceOptions.get_include_dirs = include_dirs
        try:
            print(
                json.dumps(
                    dict(
                        event="compiler_compatibility",
                        aoti_owned_constant_handles=True,
                        torch_git=TORCH_GIT,
                        header_sha256=HEADER_SHA256,
                        overlay_sha256=hashlib.sha256(fixed.encode()).hexdigest(),
                    )
                ),
                flush=True,
            )
            yield overlay
        finally:
            del CppTorchDeviceOptions.get_include_dirs
