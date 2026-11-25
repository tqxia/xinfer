"""Scoped cleanup fix for the pinned Torch 2.13 strict export capture path."""

from contextlib import contextmanager
import hashlib
from pathlib import Path

import torch

from compiler.common.artifacts import require


TORCH_GIT = "cf30153c4c131c8164ee7798e5022d810682e2cb"
CONVERT_FRAME = Path("_dynamo/convert_frame.py")
CONVERT_FRAME_SHA256 = "ca21a952feec825550c3716a95879e9196719718f18cce6238ae6d32139023b9"
ORIGINAL = """    def graph_capture_output(
        self,
        argdefs: tuple[Any, ...] | None = None,
        kwdefaults: dict[str, Any] | None = None,
    ) -> GraphCaptureOutput:"""


@contextmanager
def cleanup_export_globals():
    """Register Dynamo's installed-global cleanups on strict export capture.

    In the pinned Torch 2.13 build, ``DynamoOutput.graph_capture_output``
    omits the ``CleanupManager.instance[self.bytecode] = output_graph.cleanups``
    registration that ``_compile`` performs. Strict ``torch.export.export``
    therefore leaves ``__compiled_fn_*`` references to the exported model's
    parameters and buffers pinned in ``torch._dynamo.functional_export.__dict__``
    even after ``torch._dynamo.reset()`` and garbage collection.

    This scoped hook registers those cleanups on the captured bytecode object
    so they run as soon as Dynamo's temporary capture result goes out of scope.
    Installed Torch files are not modified. Assumes sequential exports.
    """
    from torch._dynamo import convert_frame
    from torch._dynamo.utils import CleanupManager

    require(
        str(torch.__version__).split("+", 1)[0] == "2.13.0"
        and torch.version.git_version == TORCH_GIT,
        "Strict export cleanup workaround requires the pinned Torch 2.13 build",
    )
    source_path = Path(torch.__file__).resolve().parent / CONVERT_FRAME
    source = source_path.read_text()
    require(
        hashlib.sha256(source.encode()).hexdigest() == CONVERT_FRAME_SHA256
        and source.count(ORIGINAL) == 1,
        "Unsupported Torch Dynamo convert_frame source",
    )
    original = convert_frame.DynamoOutput.graph_capture_output
    require(
        getattr(original, "__module__", None) == convert_frame.__name__
        and getattr(original, "__qualname__", None) == "DynamoOutput.graph_capture_output",
        "Unsupported Torch Dynamo graph capture hook",
    )

    def graph_capture_output(self, argdefs=None, kwdefaults=None):
        result = original(self, argdefs, kwdefaults)
        output_graph = self.tracer_output.output_graph
        if output_graph is not None and output_graph.cleanups:
            CleanupManager.instance[self.bytecode] = output_graph.cleanups
        return result

    convert_frame.DynamoOutput.graph_capture_output = graph_capture_output
    try:
        yield
    finally:
        convert_frame.DynamoOutput.graph_capture_output = original
