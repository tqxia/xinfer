"""AOT-compile Qwen3.8-27B-FP8 with the shared Qwen3.5 text pipeline."""

from compiler.qwen35.export import main as export_main
from compiler.qwen38.config import ARTIFACT_DIR, TEXT_DIR
from compiler.qwen38.model import load_model


def main(argv=None):
    export_main(
        argv,
        description=__doc__,
        model_dir=TEXT_DIR,
        output=ARTIFACT_DIR / "text",
        build_model="qwen38",
        model_loader=load_model,
    )


if __name__ == "__main__":
    main()
