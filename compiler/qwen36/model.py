"""Pristine HF Qwen3.6 FP8 model with bounded, mixed-dtype checkpoint loading."""

from collections import Counter, defaultdict
from pathlib import Path
import re

import torch
from safetensors import safe_open
from transformers import FineGrainedFP8Config, GenerationConfig, Qwen3_5MoeForCausalLM
from transformers.integrations.finegrained_fp8 import FP8Experts, replace_with_fp8_linear
from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (
    Qwen3_5MoeTextConfig,
    Qwen3_5MoeTextRotaryEmbedding,
)

from compiler.common.artifacts import MODEL_MANIFEST_NAME, read_json, require
from compiler.qwen35.model import TextCache, TextStep, state_specs, zero_state
from compiler.qwen36.config import MODEL_ID, REVISION, TEXT_DIR, configure, configure_hf_fp8

__all__ = ["load_model", "TextCache", "TextStep", "state_specs", "zero_state"]

_EXPERT = re.compile(r"(model\.layers\.\d+\.mlp\.experts)\.\d+\.(gate|up|down)_proj\.")


def _configuration(document, num_hidden_layers):
    text, quant = document["text_config"], document["quantization_config"]
    require(
        document["model_type"] == "qwen3_5_moe"
        and text["model_type"] == "qwen3_5_moe_text"
        and text["tie_word_embeddings"] is False
        and document["tie_word_embeddings"] is False
        and text["rope_parameters"]["rope_type"] == "default"
        and text["hidden_act"] == "silu"
        and text["attention_bias"] is False
        and text["attn_output_gate"] is True
        and text.get("swiglu_alpha") is None
        and text.get("swiglu_limit") is None,
        "Qwen3.6 loading requires the untied, bias-free gated MoE architecture with default RoPE",
    )
    require(
        quant["quant_method"] == "fp8"
        and quant["fmt"] == "e4m3"
        and quant["activation_scheme"] == "dynamic"
        and quant["weight_block_size"] == [128, 128],
        "Qwen3.6 loading requires dynamic E4M3 FP8 with 128x128 blocks",
    )
    config = Qwen3_5MoeTextConfig.from_dict(text)
    original_layers = config.num_hidden_layers
    layers = original_layers if num_hidden_layers is None else num_hidden_layers
    require(type(layers) is int and 1 <= layers <= original_layers, "Invalid decoder prefix length")
    config.num_hidden_layers = layers
    config.layer_types = config.layer_types[:layers]
    config._attn_implementation = "sdpa"
    return config, original_layers


def _selected(name, layers):
    if name.startswith("model.layers."):
        return int(name.split(".")[2]) < layers
    return name in ("model.embed_tokens.weight", "model.norm.weight", "lm_head.weight")


def _read(reader, name, inventory, consumed):
    require(name not in consumed, f"Checkpoint tensor was consumed twice: {name}")
    value, record = reader.get_tensor(name), inventory[name]
    require(
        record["dtype"] in ("bfloat16", "float8_e4m3fn")
        and str(value.dtype) == "torch." + record["dtype"]
        and list(value.shape) == record["shape"]
        and value.numel() * value.element_size() == record["bytes"],
        f"Checkpoint dtype, shape or byte count differs: {name}",
    )
    # Check source values in bounded CPU chunks, never a dequantized bank.
    for part in value.reshape(-1).split(1024 * 1024):
        require(bool(torch.isfinite(part.float()).all()), f"Nonfinite checkpoint tensor: {name}")
    consumed.add(name)
    return value


def _assign(model, name, value, device, assigned):
    require(name not in assigned, f"Model parameter was assigned twice: {name}")
    destination = model.get_parameter(name)
    expected_dtype = (
        torch.float8_e4m3fn if destination.dtype == torch.float8_e4m3fn else torch.bfloat16
    )
    require(
        value.shape == destination.shape and value.dtype == expected_dtype,
        f"Parameter type or shape differs: {name}",
    )
    parent, _, field = name.rpartition(".")
    # No dtype argument: retain FP8 weights and the checkpoint's BF16 scales.
    model.get_submodule(parent).register_parameter(
        field, torch.nn.Parameter(value.to(device=device), requires_grad=False)
    )
    assigned.add(name)


def _pack_experts(model, prefix, reader, names, inventory, device, assigned, consumed):
    experts = model.get_submodule(prefix)
    require(
        isinstance(experts, FP8Experts) and experts.has_gate and not experts.has_bias,
        "Expected official gated FP8 experts",
    )
    expected = {
        f"{prefix}.{expert}.{projection}.{suffix}"
        for expert in range(experts.num_experts)
        for projection in ("gate_proj", "up_proj", "down_proj")
        for suffix in ("weight", "weight_scale_inv")
    }
    require(names == expected, f"Incomplete or split expert source shard: {prefix}")
    maximum = 0
    for target, projections, suffix, dtype in (
        ("gate_up_proj", ("gate_proj", "up_proj"), "weight", torch.float8_e4m3fn),
        ("gate_up_proj_scale_inv", ("gate_proj", "up_proj"), "weight_scale_inv", torch.bfloat16),
        ("down_proj", ("down_proj",), "weight", torch.float8_e4m3fn),
        ("down_proj_scale_inv", ("down_proj",), "weight_scale_inv", torch.bfloat16),
    ):
        shape = getattr(experts, target).shape
        packed = torch.empty(shape, dtype=dtype, device="cpu")
        rows = shape[1] // len(projections)
        for expert in range(experts.num_experts):
            for slot, projection in enumerate(projections):
                name = f"{prefix}.{expert}.{projection}.{suffix}"
                value = _read(reader, name, inventory, consumed)
                destination = packed[expert, slot * rows : (slot + 1) * rows]
                require(
                    value.shape == destination.shape and value.dtype == dtype,
                    f"Expert packing differs: {name}",
                )
                destination.copy_(value)
        maximum = max(maximum, packed.numel() * packed.element_size())
        _assign(model, prefix + "." + target, packed, device, assigned)
        del packed
    return maximum


def load_model(directory=TEXT_DIR, device="cuda", *, num_hidden_layers=None):
    """Load a verified prepared checkpoint, optionally selecting a decoder prefix.

    Preparation/explicit verification owns file hashes. Loading checks typed
    inventory, finite values and complete materialization without rehashing the
    checkpoint. CPU loading is intended for small fixtures and inspection.
    """
    configure()
    directory, device = Path(directory), torch.device(device)
    require(device.type in ("cpu", "cuda"), "Qwen3.6 loading requires CPU or CUDA")
    document = read_json(directory / "config.json")
    config, original_layers = _configuration(document, num_hidden_layers)
    manifest = read_json(directory / MODEL_MANIFEST_NAME)
    require(
        manifest["model"] == MODEL_ID and manifest["revision"] == REVISION,
        "Unsupported prepared checkpoint identity",
    )
    weight_map = read_json(directory / "text.safetensors.index.json")["weight_map"]
    ledger = read_json(directory / "text_inventory.json")
    inventory, skiplist = ledger["tensors"], ledger["prepared_modules_to_not_convert"]
    translated = [
        "model." + name.removeprefix("model.language_model.")
        if name.startswith("model.language_model.")
        else name
        for name in document["quantization_config"]["modules_to_not_convert"]
    ]
    require(
        skiplist == translated and set(weight_map) == set(inventory),
        "Prepared quantization names or tensor inventory differ",
    )
    require(
        len(inventory) == manifest["tensor_count"]
        and sum(value["bytes"] for value in inventory.values()) == manifest["tensor_bytes"],
        "Prepared source tensor budget differs",
    )

    with torch.device("meta"):
        model = Qwen3_5MoeForCausalLM(config)
        replace_with_fp8_linear(
            model,
            modules_to_not_convert=skiplist,
            quantization_config=FineGrainedFP8Config(
                activation_scheme="dynamic", weight_block_size=(128, 128), dequantize=False
            ),
            pre_quantized=True,
        )
    require(config._experts_implementation == "grouped_mm", "HF default expert backend changed")
    expected_parameters = set(dict(model.named_parameters()))
    selected = {
        name: shard
        for name, shard in weight_map.items()
        if _selected(name, config.num_hidden_layers)
    }
    by_shard = defaultdict(set)
    for name, shard in selected.items():
        require(
            Path(shard).name == shard and shard.endswith(".safetensors"),
            "Invalid prepared shard path",
        )
        by_shard[shard].add(name)
    assigned, consumed, maximum = set(), set(), 0
    for shard, names in sorted(by_shard.items()):
        with safe_open(directory / shard, framework="pt", device="cpu") as reader:
            groups = defaultdict(set)
            ordinary = []
            for name in sorted(names):
                match = _EXPERT.match(name)
                if match:
                    groups[match[1]].add(name)
                else:
                    ordinary.append(name)
            for prefix, sources in sorted(groups.items()):
                maximum = max(
                    maximum,
                    _pack_experts(
                        model, prefix, reader, sources, inventory, device, assigned, consumed
                    ),
                )
            for name in ordinary:
                value = _read(reader, name, inventory, consumed)
                _assign(model, name, value, device, assigned)
    require(
        consumed == set(selected) and assigned == expected_parameters,
        "Checkpoint did not completely materialize the HF model",
    )
    # Nonpersistent buffers created on meta are not checkpoint parameters.
    model.model.rotary_emb = Qwen3_5MoeTextRotaryEmbedding(config).to(device=device)
    model.generation_config = GenerationConfig.from_dict(
        read_json(directory / "generation_config.json")
    )
    require(
        len(model.generation_config.eos_token_id) == 2
        and config.eos_token_id in model.generation_config.eos_token_id,
        "Generation EOS contract differs",
    )
    require(
        model.lm_head.weight.data_ptr() != model.model.embed_tokens.weight.data_ptr(),
        "The checkpoint LM head must remain untied",
    )
    require(
        all(not value.is_meta for value in (*model.parameters(), *model.buffers())),
        "Unmaterialized meta tensors remain",
    )
    dtype_bytes = Counter()
    for value in model.parameters():
        dtype_bytes[str(value.dtype).removeprefix("torch.")] += value.numel() * value.element_size()
    source_bytes = sum(inventory[name]["bytes"] for name in selected)
    require(
        sum(dtype_bytes.values()) == source_bytes,
        "Packed model storage differs from source tensor budget",
    )
    model.checkpoint_storage = dict(
        checkpoint_layers=original_layers,
        loaded_layers=config.num_hidden_layers,
        checkpoint_source_tensors=len(inventory),
        loaded_source_tensors=len(consumed),
        source_bytes=source_bytes,
        parameter_bytes=sum(dtype_bytes.values()),
        parameter_bytes_by_dtype=dict(dtype_bytes),
        max_cpu_packed_bytes=maximum,
    )
    if device.type == "cuda":
        configure_hf_fp8()
    return model.eval().requires_grad_(False)
