"""Export Qwen prefill and decode as one shared-weight AOTInductor bundle."""

from functools import partial
import time
from pathlib import Path

import torch

from compiler.common.artifacts import MODEL_MANIFEST_NAME, read_json
from compiler.common.attention import cache_extent, attention_library, attention_c_shims
from compiler.common.export import (
    ExportInputError,
    check_symbolic_config,
    export_parser,
    prepare_export,
    report_export,
)
from compiler.common.weights import compile_shared
from compiler.common.metadata import make_metadata, state_spec
from compiler.qwen3.adapters.qwen3_flash import FlashStep
from compiler.qwen3.model import MAX_PREFILL, allocate_cache, load_model
from compiler.qwen3.config import DEFAULT_MODEL


def example_inputs(config, kind, capacity):
    length = 1 if kind == "decode" else 16
    cache = allocate_cache(config, capacity)
    inputs = (
        torch.ones((1, length), dtype=torch.int64, device="cuda"),
        torch.arange(length, device="cuda"),
        cache,
        cache_extent(32),
    )
    token_shape = {}
    if kind == "prefill":
        seq = torch.export.Dim("tokens", min=2, max=MAX_PREFILL)
        token_shape = {1: seq}
    extent = torch.export.Dim("kv_extent", min=2, max=capacity + 1)
    shapes = (
        token_shape,
        {} if kind == "decode" else {0: seq},
        tuple({} for _ in cache),
        {0: extent},
    )
    return inputs, shapes


def export_step(step, inputs, shapes):
    exported = torch.export.export(step, inputs, dynamic_shapes=shapes, strict=False)
    exported.validate()
    return exported


def check_graph(exported, layers):
    targets = [n.target for n in exported.graph.nodes if n.op == "call_function"]
    sites = targets.count(torch.ops.xinfer.flash_attention.default)
    updates = targets.count(torch.ops.aten.index_copy_.default)
    assert sites == layers and updates == 2 * sites
    assert torch.ops.aten.scaled_dot_product_attention.default not in targets
    return {"name": "qwen3_flash_v1", "replacements": sites, "cache_updates": updates}


def entry_metadata(model, kind, checkpoint, capacity, cache_shape):
    from torch._inductor.runtime.triton_helpers import get_backend_options

    state = [
        state_spec(layer, name, list(cache_shape), "bfloat16", "inplace")
        for layer in range(model.config.num_hidden_layers)
        for name in ("key", "value")
    ]
    return make_metadata(
        model_type="qwen3",
        entry_point=kind,
        token_range=[1, 1] if kind == "decode" else [2, MAX_PREFILL],
        checkpoint=checkpoint,
        torch_version=torch.__version__,
        cxx11_abi=torch._C._GLIBCXX_USE_CXX11_ABI,
        cuda_compute_capability=list(torch.cuda.get_device_capability()),
        cache_length=capacity,
        vocab_size=model.config.vocab_size,
        eos_token_ids=model.generation_config.eos_token_id,
        state=state,
        triton_enable_fp_fusion=get_backend_options()["enable_fp_fusion"],
        model_specific={},
    )


def export_entry(kind, staging_directory, *, model, step, checkpoint, capacity):
    inputs, shapes = example_inputs(model.config, kind, capacity)
    start = time.perf_counter()
    exported = export_step(step, inputs, shapes)
    adapter_report = check_graph(exported, model.config.num_hidden_layers)
    # Text dumps expose mutation/shape behavior without another copy of the weights.
    (staging_directory / f"{kind}.graph.txt").write_text(exported.graph_module.code)
    print(f"Exported {kind}; compiling...", flush=True)

    def metadata_factory(_compiled_path):
        metadata = entry_metadata(model, kind, checkpoint, capacity, inputs[2][0].shape)
        report_export(
            "qwen3",
            kind,
            strict=False,
            diagnostics=dict(
                graph_nodes=len(list(exported.graph.nodes)),
                export_and_compile_seconds=time.perf_counter() - start,
                model_adapter=adapter_report,
            ),
        )
        return metadata

    return exported, metadata_factory


def export_model(*, model_dir, output, max_context_length=None):
    """Load one adapted model and publish its paired shared-weight bundle."""
    model_dir, output = Path(model_dir), Path(output)
    capacity = prepare_export(model_dir, output, max_context_length, MAX_PREFILL)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    check_symbolic_config()
    model = load_model(model_dir)
    step = FlashStep(model)
    manifest = read_json(model_dir / MODEL_MANIFEST_NAME)
    revision = manifest["revision"]
    checkpoint = dict(model_id=manifest["repo_id"], revision=revision, manifest=manifest)
    # Retain BF16 cast boundaries; fused arithmetic can still differ from eager.
    options = {"emulate_precision_casts": True}
    prepare_entry = partial(
        export_entry, model=model, step=step, checkpoint=checkpoint, capacity=capacity
    )
    with torch.inference_mode():
        compile_shared(
            output,
            prepare_entry,
            identity=dict(model="qwen3-0.6b", revision=revision),
            inductor_configs=options,
            native_libraries=[attention_library()],
            custom_op_shims=attention_c_shims(),
        )
    print(f"Saved prefill/decode bundle to {output}", flush=True)


def main(argv=None):
    parser = export_parser(__doc__, model_dir=DEFAULT_MODEL, output=Path("artifacts/aoti"))
    args = parser.parse_args(argv)
    try:
        export_model(**vars(args))
    except ExportInputError as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
