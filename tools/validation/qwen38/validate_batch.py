"""Validate Qwen3.8 batches through FP8 HF, packed eager, AOTI and C++."""

from tools.validation.qwen35.validate_batch import main as shared_main


def main(argv=None):
    shared_main(argv, description=__doc__, expected_model_type="qwen38")


if __name__ == "__main__":
    main()
