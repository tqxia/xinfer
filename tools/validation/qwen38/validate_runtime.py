"""Validate Qwen3.8 native runtime behavior and package contracts."""

from compiler.qwen38.config import ARTIFACT_DIR, TEXT_DIR
from tools.validation.qwen35.validate_runtime import main as shared_main


def main(argv=None):
    shared_main(
        argv,
        model_dir=TEXT_DIR,
        artifact_dir=ARTIFACT_DIR,
        expected_model_type="qwen38",
        description=__doc__,
    )


if __name__ == "__main__":
    main()
