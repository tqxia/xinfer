"""Qwen3.8 FP8 preparation through the shared extraction and verification pipeline."""

from tools.prepare import qwen38
from tools.prepare.tests import test_qwen35


class PreparedCheckpoint(test_qwen35.PreparedCheckpoint):
    frontend = qwen38
    model_id = "Qwen/Qwen3.8-27B-FP8"
