"""Validate Qwen3 attention packages and load their native dependency."""

from compiler.common.attention import OPERATOR_CONTRACT, load_attention_ops, native_attention_library


def load_package_attention_ops(metadata, native=None):
    if 'model_type' in metadata and metadata['model_type'] != 'qwen3':
        raise ValueError('Unsupported model type')
    if metadata.get('attention_backend', 'efficient') != 'flash':
        raise ValueError('Legacy efficient packages are unsupported; re-export Qwen3 packages with the current exporter')
    if metadata.get('eager_numerics', False):
        raise ValueError('Legacy eager-numerics packages are unsupported; re-export Qwen3 packages with the current exporter')
    if 'custom_attention' not in metadata:
        raise ValueError('Flash packages require the custom-op contract; re-export Qwen3 packages with the current exporter')
    if metadata['custom_attention'] != OPERATOR_CONTRACT:
        raise ValueError('Unsupported custom attention contract')
    library = native_attention_library(native) if native is not None else None
    if native is not None and library is None:
        raise ValueError('Qwen3 runtime is missing its required native attention library; rebuild the runtime')
    load_attention_ops(library)
