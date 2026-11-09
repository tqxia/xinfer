"""Prepare pinned Qwen3.6 FP8 text shards without changing stored tensors."""

from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import re

from compiler.common.artifacts import MODEL_MANIFEST_NAME, json_sha256, require, sha256, write_json
from tools.prepare import common

MODEL_ID = "Qwen/Qwen3.6-35B-A3B-FP8"
REVISIONS = {MODEL_ID: "c653bba46dfa06e9658dcc0b0dc87fe4685f14a2"}
TEXT_DIR = Path("work/qwen36/35b-a3b-fp8/text")
MODEL_FILES = (
    "config.json",
    "generation_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "chat_template.jinja",
)
INDEX_NAME = "text.safetensors.index.json"
INVENTORY_NAME = "text_inventory.json"
SOURCE_INDEX_NAME = "source.safetensors.index.json"
PREFIX = "model.language_model."
ELEMENT_BYTES = {"bfloat16": 2, "float8_e4m3fn": 1}


def default_model_dir(model_id):
    require(
        isinstance(model_id, str) and model_id in REVISIONS, f"Unsupported checkpoint: {model_id}"
    )
    return TEXT_DIR


def _prepared_name(name):
    return "model." + name.removeprefix(PREFIX) if name.startswith(PREFIX) else name


def _skiplist(config):
    original = config["quantization_config"]["modules_to_not_convert"]
    require(
        isinstance(original, list) and all(isinstance(name, str) and name for name in original),
        "Invalid FP8 module exclusions",
    )
    # Retain intentionally unused vision/MTP and legacy entries; only rename
    # the text prefix. Future loading uses this list, not the original names.
    return [_prepared_name(name) for name in original]


def _generation(config, generation):
    eos = generation.get("eos_token_id")
    require(
        isinstance(eos, list)
        and len(eos) == 2
        and all(
            type(token) is int and 0 <= token < config["text_config"]["vocab_size"] for token in eos
        )
        and len(set(eos)) == 2
        and config["text_config"]["eos_token_id"] in eos,
        "Expected the checkpoint's two distinct valid EOS tokens",
    )


def _expected(config):
    """Derive the complete original text inventory from the pinned architecture."""
    text, quant = config["text_config"], config["quantization_config"]
    require(
        config["model_type"] == "qwen3_5_moe"
        and text["model_type"] == "qwen3_5_moe_text"
        and text["tie_word_embeddings"] is False
        and config["tie_word_embeddings"] is False
        and text["rope_parameters"]["rope_type"] == "default"
        and text["hidden_act"] == "silu"
        and text["attention_bias"] is False
        and text["attn_output_gate"] is True,
        "Expected the untied Qwen3.6 text MoE architecture with default RoPE",
    )
    require(
        quant["quant_method"] == "fp8"
        and quant["fmt"] == "e4m3"
        and quant["activation_scheme"] == "dynamic"
        and quant["weight_block_size"] == [128, 128],
        "Expected dynamic E4M3 FP8 with 128x128 weight blocks",
    )
    _skiplist(config)
    fields = (
        "hidden_size",
        "vocab_size",
        "num_hidden_layers",
        "num_experts",
        "num_experts_per_tok",
        "moe_intermediate_size",
        "shared_expert_intermediate_size",
        "num_attention_heads",
        "num_key_value_heads",
        "head_dim",
        "linear_num_key_heads",
        "linear_num_value_heads",
        "linear_key_head_dim",
        "linear_value_head_dim",
        "linear_conv_kernel_dim",
    )
    require(
        all(type(text[key]) is int and text[key] > 0 for key in fields), "Invalid model geometry"
    )
    require(
        len(text["layer_types"]) == text["num_hidden_layers"]
        and text["num_experts_per_tok"] <= text["num_experts"]
        and text["num_attention_heads"] % text["num_key_value_heads"] == 0
        and text["linear_num_value_heads"] % text["linear_num_key_heads"] == 0,
        "Inconsistent layer, expert or attention geometry",
    )
    hidden, expected = text["hidden_size"], {}

    def bf16(name, shape):
        expected[name] = dict(dtype="bfloat16", shape=list(shape))

    def fp8(name, rows, columns):
        require(rows % 128 == 0 and columns % 128 == 0, f"Unaligned FP8 projection: {name}")
        expected[name + ".weight"] = dict(dtype="float8_e4m3fn", shape=[rows, columns])
        bf16(name + ".weight_scale_inv", [rows // 128, columns // 128])

    bf16(PREFIX + "embed_tokens.weight", [text["vocab_size"], hidden])
    bf16(PREFIX + "norm.weight", [hidden])
    bf16("lm_head.weight", [text["vocab_size"], hidden])
    for layer, kind in enumerate(text["layer_types"]):
        base = PREFIX + f"layers.{layer}."
        bf16(base + "input_layernorm.weight", [hidden])
        bf16(base + "post_attention_layernorm.weight", [hidden])
        bf16(base + "mlp.gate.weight", [text["num_experts"], hidden])
        bf16(base + "mlp.shared_expert_gate.weight", [1, hidden])
        for expert in range(text["num_experts"]):
            name = base + f"mlp.experts.{expert}."
            inner = text["moe_intermediate_size"]
            fp8(name + "gate_proj", inner, hidden)
            fp8(name + "up_proj", inner, hidden)
            fp8(name + "down_proj", hidden, inner)
        inner = text["shared_expert_intermediate_size"]
        fp8(base + "mlp.shared_expert.gate_proj", inner, hidden)
        fp8(base + "mlp.shared_expert.up_proj", inner, hidden)
        fp8(base + "mlp.shared_expert.down_proj", hidden, inner)
        if kind == "full_attention":
            dim = text["head_dim"]
            query, kv = text["num_attention_heads"] * dim, text["num_key_value_heads"] * dim
            fp8(base + "self_attn.q_proj", 2 * query, hidden)
            fp8(base + "self_attn.k_proj", kv, hidden)
            fp8(base + "self_attn.v_proj", kv, hidden)
            fp8(base + "self_attn.o_proj", hidden, query)
            for norm in ("q_norm", "k_norm"):
                bf16(base + f"self_attn.{norm}.weight", [dim])
        else:
            require(kind == "linear_attention", f"Unknown layer type: {kind}")
            heads = text["linear_num_value_heads"]
            value = heads * text["linear_value_head_dim"]
            qkv = 2 * text["linear_num_key_heads"] * text["linear_key_head_dim"] + value
            bf16(base + "linear_attn.conv1d.weight", [qkv, 1, text["linear_conv_kernel_dim"]])
            bf16(base + "linear_attn.norm.weight", [text["linear_value_head_dim"]])
            for name in ("A_log", "dt_bias"):
                bf16(base + "linear_attn." + name, [heads])
            for name in ("in_proj_a", "in_proj_b"):
                bf16(base + f"linear_attn.{name}.weight", [heads, hidden])
            fp8(base + "linear_attn.in_proj_qkv", qkv, hidden)
            fp8(base + "linear_attn.in_proj_z", value, hidden)
            fp8(base + "linear_attn.out_proj", hidden, value)
    return expected


def _descriptor(value):
    return dict(dtype=str(value.dtype).removeprefix("torch."), shape=list(value.shape))


def _tensor_hash(value):
    import torch

    require(value.device.type == "cpu" and value.is_contiguous(), "Expected contiguous CPU tensors")
    return hashlib.sha256(memoryview(value.view(torch.uint8).numpy()).cast("B")).hexdigest()


def _summary(tensors):
    counts, sizes = Counter(), Counter()
    for value in tensors.values():
        counts[value["dtype"]] += 1
        sizes[value["dtype"]] += value["bytes"]
    return dict(
        tensor_count=sum(counts.values()),
        tensor_bytes=sum(sizes.values()),
        dtype_inventory={
            name: dict(tensors=counts[name], bytes=sizes[name]) for name in sorted(counts)
        },
    )


def _mapping_hash(tensors):
    return json_sha256({value["source"]: name for name, value in tensors.items()})


def _extract(source, destination, names, expected):
    """Hold tensor views for one source shard only; retain no full-model weight dictionary."""
    from safetensors import safe_open
    from safetensors.torch import save_file

    weights, inventory = {}, {}
    with safe_open(source, framework="pt", device="cpu") as reader:
        for name in sorted(names):
            value = reader.get_tensor(name)
            require(_descriptor(value) == expected[name], f"Tensor dtype or shape differs: {name}")
            prepared = _prepared_name(name)
            weights[prepared] = value
            inventory[prepared] = dict(
                **expected[name],
                source=name,
                source_shard=source.name,
                bytes=value.numel() * value.element_size(),
                sha256=_tensor_hash(value),
            )
        save_file(weights, destination)
    with safe_open(destination, framework="pt", device="cpu") as reader:
        require(set(reader.keys()) == set(inventory), "Prepared shard inventory differs")
        for name, record in inventory.items():
            value = reader.get_tensor(name)
            require(
                _descriptor(value) == {key: record[key] for key in ("dtype", "shape")}
                and _tensor_hash(value) == record["sha256"],
                f"Prepared tensor bytes differ: {name}",
            )
    return inventory


def prepare(directory, model=MODEL_ID, cache_dir=None):
    """Atomically publish verified text-only shards, preserving every stored tensor byte."""
    default_model_dir(model)
    with common.checkpoint_directory(directory) as staging:
        hub = common.Hub(model, REVISIONS[model], cache_dir=cache_dir)
        for name in MODEL_FILES:
            hub.copy(name, staging / name)
        config = json.loads((staging / "config.json").read_text())
        expected = _expected(config)
        generation = json.loads((staging / "generation_config.json").read_text())
        _generation(config, generation)
        hub.copy("model.safetensors.index.json", staging / SOURCE_INDEX_NAME)
        index = json.loads((staging / SOURCE_INDEX_NAME).read_text())["weight_map"]
        selected = {
            name: shard
            for name, shard in index.items()
            if name.startswith(PREFIX) or name.startswith("lm_head.")
        }
        require(
            set(selected) == set(expected),
            "Complete text tensor inventory differs from model config",
        )
        by_shard = defaultdict(list)
        for name, shard in selected.items():
            by_shard[shard].append(name)
        inventory, weight_map, files = {}, {}, {}
        for number, shard in enumerate(sorted(by_shard), 1):
            source = hub.fetch(shard)
            filename = f"text-{number:05d}-of-{len(by_shard):05d}.safetensors"
            records = _extract(source, staging / filename, by_shard[shard], expected)
            # The cache basename may be a content hash rather than the source shard name.
            for record in records.values():
                record["source_shard"] = shard
            inventory.update(records)
            weight_map.update({name: filename for name in records})
            files[filename] = sha256(staging / filename)
        summary = _summary(inventory)
        write_json(
            staging / INDEX_NAME,
            dict(metadata=dict(total_size=summary["tensor_bytes"]), weight_map=weight_map),
        )
        write_json(
            staging / INVENTORY_NAME,
            dict(format=1, prepared_modules_to_not_convert=_skiplist(config), tensors=inventory),
        )
        files.update(
            {
                name: sha256(staging / name)
                for name in (*MODEL_FILES, INDEX_NAME, INVENTORY_NAME, SOURCE_INDEX_NAME)
            }
        )
        manifest = dict(
            format=1,
            model=model,
            revision=REVISIONS[model],
            **summary,
            source_to_prepared_sha256=_mapping_hash(inventory),
            source_files=dict(sorted(hub.verified.items())),
            files=dict(sorted(files.items())),
        )
        write_json(staging / MODEL_MANIFEST_NAME, manifest)
    return manifest


def verify_model(directory):
    """Verify the complete prepared checkpoint locally, without Hub access or inference."""
    from safetensors import safe_open

    directory = Path(directory)
    manifest = json.loads((directory / MODEL_MANIFEST_NAME).read_text())
    require(
        isinstance(manifest, dict)
        and manifest.get("format") == 1
        and isinstance(manifest.get("model"), str)
        and manifest["model"] in REVISIONS
        and manifest.get("revision") == REVISIONS[manifest["model"]],
        "Prepared checkpoint must match a supported pinned revision",
    )
    files = manifest.get("files")
    fixed = {*MODEL_FILES, INDEX_NAME, INVENTORY_NAME, SOURCE_INDEX_NAME}
    require(isinstance(files, dict) and fixed <= set(files), "Prepared file manifest is incomplete")
    shards = set(files) - fixed
    expected_shards = {
        f"text-{number:05d}-of-{len(shards):05d}.safetensors"
        for number in range(1, len(shards) + 1)
    }
    require(shards and shards == expected_shards, "Prepared shard file manifest is invalid")
    for name, digest in files.items():
        require(
            isinstance(digest, str) and re.fullmatch("[0-9a-f]{64}", digest),
            "Invalid file checksum",
        )
        require(sha256(directory / name) == digest, f"Checksum mismatch: {name}")
    config = json.loads((directory / "config.json").read_text())
    _generation(config, json.loads((directory / "generation_config.json").read_text()))
    originals = _expected(config)
    expected = {_prepared_name(name): value for name, value in originals.items()}
    inventory = json.loads((directory / INVENTORY_NAME).read_text())
    require(
        inventory.get("format") == 1 and isinstance(inventory.get("tensors"), dict),
        "Invalid tensor inventory",
    )
    require(
        inventory.get("prepared_modules_to_not_convert") == _skiplist(config),
        "Prepared FP8 module exclusions differ",
    )
    tensors = inventory["tensors"]
    source_files = manifest.get("source_files")
    require(isinstance(source_files, dict), "Invalid original source file inventory")
    for name, descriptor in source_files.items():
        require(
            isinstance(name, str)
            and name
            and not Path(name).is_absolute()
            and ".." not in Path(name).parts
            and isinstance(descriptor, dict)
            and set(descriptor) == {"bytes", "sha256"}
            and type(descriptor["bytes"]) is int
            and descriptor["bytes"] >= 0
            and isinstance(descriptor["sha256"], str)
            and re.fullmatch("[0-9a-f]{64}", descriptor["sha256"]),
            "Invalid original source file descriptor",
        )
    for prepared, source in [(name, name) for name in MODEL_FILES] + [
        (SOURCE_INDEX_NAME, "model.safetensors.index.json")
    ]:
        require(
            source in source_files
            and files[prepared] == source_files[source]["sha256"]
            and (directory / prepared).stat().st_size == source_files[source]["bytes"],
            f"Copied source file differs: {prepared}",
        )
    original_index = json.loads((directory / SOURCE_INDEX_NAME).read_text())["weight_map"]
    selected = {
        name: shard
        for name, shard in original_index.items()
        if name.startswith(PREFIX) or name.startswith("lm_head.")
    }
    require(set(selected) == set(originals), "Original text inventory differs")
    require(
        set(source_files) == {*MODEL_FILES, "model.safetensors.index.json", *selected.values()},
        "Original source file inventory differs",
    )
    index = json.loads((directory / INDEX_NAME).read_text())
    weight_map = index["weight_map"]
    require(
        set(tensors) == set(expected) == set(weight_map), "Indexed text tensor inventory differs"
    )
    require(set(weight_map.values()) == shards, "Indexed shard inventory differs")
    for shard in sorted(shards):
        names = {name for name, filename in weight_map.items() if filename == shard}
        with safe_open(directory / shard, framework="pt", device="cpu") as reader:
            require(set(reader.keys()) == names, "Prepared shard tensor inventory differs")
            for name in names:
                value, record = reader.get_tensor(name), tensors[name]
                actual = _descriptor(value)
                require(actual == expected[name], f"Prepared tensor dtype or shape differs: {name}")
                require(
                    isinstance(record, dict)
                    and set(record)
                    == {"source", "source_shard", "dtype", "shape", "bytes", "sha256"}
                    and all(record[key] == expected[name][key] for key in ("dtype", "shape"))
                    and _prepared_name(record["source"]) == name
                    and record["source_shard"] == selected.get(record["source"])
                    and record["bytes"]
                    == math.prod(record["shape"]) * ELEMENT_BYTES[record["dtype"]]
                    and isinstance(record["sha256"], str)
                    and re.fullmatch("[0-9a-f]{64}", record["sha256"]),
                    f"Prepared tensor provenance differs: {name}",
                )
                require(
                    _tensor_hash(value) == record["sha256"],
                    f"Prepared tensor checksum differs: {name}",
                )
    summary = _summary(tensors)
    require(
        all(manifest[key] == value for key, value in summary.items()),
        "Tensor inventory totals differ",
    )
    require(
        index["metadata"]["total_size"] == summary["tensor_bytes"], "Indexed tensor bytes differ"
    )
    require(
        manifest["source_to_prepared_sha256"] == _mapping_hash(tensors),
        "Tensor name mapping differs",
    )
    return manifest
