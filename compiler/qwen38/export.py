"""AOT-compile Qwen3.8-27B-FP8 with the shared Qwen3.5 text pipeline."""

from compiler.common.adapters.finegrained_fp8 import install_fp8
from compiler.qwen35.export import export_step as shared_export_step
from compiler.qwen35.export import main as export_main
from compiler.qwen38.config import ARTIFACT_DIR, TEXT_DIR
from compiler.qwen38.model import load_model


def export_step(model, kind, capacity=None):
    """Adapt FP8 projections before the shared graph export and packing passes."""
    install_fp8(model)
    return shared_export_step(model, kind, capacity)


def main(argv=None):
    export_main(
        argv,
        description=__doc__,
        model_dir=TEXT_DIR,
        output=ARTIFACT_DIR / "text",
        model_type="qwen38",
        model_loader=load_model,
        step_exporter=export_step,
    )


if __name__ == "__main__":
    main()
