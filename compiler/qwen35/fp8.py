"""Load HF FP8 projections unchanged, then adapt their launches for strict export."""

import torch
from safetensors import safe_open
from transformers.integrations import finegrained_fp8 as hf_fp8

from compiler.common.artifacts import require
from compiler.common.kernels.fp8 import linear, load_fp8_kernels


def validate_quantization(config):
    require(
        config.get("quant_method") == "fp8"
        and config.get("fmt") == "e4m3"
        and config.get("activation_scheme") == "dynamic"
        and config.get("weight_block_size") == [128, 128],
        "Expected dynamic E4M3 FP8 with 128x128 weight blocks",
    )


def load_weights(model, path):
    """Assign prepared tensors directly to CUDA, preserving FP8 and BF16 dtypes."""
    with safe_open(path, framework="pt", device="cpu") as reader:
        names = set(reader.keys())
        for name in sorted(names):
            if not name.endswith(".weight_scale_inv"):
                continue
            module_name = name.removesuffix(".weight_scale_inv")
            original = model.get_submodule(module_name)
            require(
                type(original) is torch.nn.Linear and original.bias is None,
                f"Expected a bias-free FP8 projection: {module_name}",
            )
            with torch.device("meta"):
                replacement = hf_fp8.FP8Linear(
                    original.in_features,
                    original.out_features,
                    block_size=(128, 128),
                    activation_scheme="dynamic",
                    has_bias=False,
                )
            # Keep HF eager on the selected kernels-community implementation,
            # including environments where optional DeepGEMM is available.
            replacement._deepgemm_disabled = True
            model.set_submodule(module_name, replacement)
        expected = model.state_dict()
        require(names == set(expected), "FP8 checkpoint/model keys differ")
        for name in sorted(names):
            value = reader.get_tensor(name)
            require(
                value.shape == expected[name].shape
                and value.dtype in (torch.bfloat16, torch.float8_e4m3fn)
                and (value.dtype == torch.float8_e4m3fn)
                == (expected[name].dtype == torch.float8_e4m3fn),
                f"FP8 checkpoint shape/dtype mismatch: {name}",
            )
            parent, _, field = name.rpartition(".")
            model.get_submodule(parent).register_parameter(
                field, torch.nn.Parameter(value.to("cuda"), requires_grad=False)
            )

    # Resolve the official HF backend to the same loaded upstream build used
    # by export. Restore the resolver immediately; HF forward math is unchanged.
    kernel = load_fp8_kernels()
    resolver = hf_fp8.lazy_load_kernel
    hf_fp8.lazy_load_kernel = lambda name: kernel if name == "finegrained-fp8" else resolver(name)
    try:
        loaded = hf_fp8.load_finegrained_fp8_kernel()
        require(loaded.matmul is kernel.matmul_2d, "HF already loaded a different FP8 kernel build")
    finally:
        hf_fp8.lazy_load_kernel = resolver


class FP8Linear(torch.nn.Module):
    def __init__(self, original):
        super().__init__()
        self.weight = original.weight
        self.weight_scale_inv = original.weight_scale_inv

    def forward(self, x):
        return linear(x.contiguous(), self.weight, self.weight_scale_inv)


def install_fp8(model):
    """Reuse HF parameters and upstream kernel math with symbolic launch dimensions."""
    for name, module in list(model.named_modules()):
        if type(module) is hf_fp8.FP8Linear:
            model.set_submodule(name, FP8Linear(module))
