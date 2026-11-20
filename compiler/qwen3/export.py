"""Export Qwen prefill and decode as one shared-weight AOTInductor bundle."""

from functools import partial
import time
from pathlib import Path

import torch
from torch.fx.experimental import _config as symbolic_config

from compiler.common.artifacts import MODEL_MANIFEST_NAME, read_json
from compiler.common.attention import (
    attention_library,
    attention_c_shims,
)
from compiler.common.export import (
    ExportInputError,
    export_parser,
    prepare_export,
    report_export,
)
from compiler.common.weights import compile_shared
from compiler.common.metadata import make_metadata, state_spec
from compiler.qwen3.model import MAX_BATCH_SIZE, MAX_PREFILL, FlashStep, allocate_cache, load_model
from compiler.qwen3.config import DEFAULT_MODEL


def example_inputs(config, kind, capacity):
    batch_size = 2
    length = 1 if kind == "decode" else 16
    cache = allocate_cache(config, capacity, batch_size)
    starts = [16 + 8 * row for row in range(batch_size)]
    counts = [max(1, length // (row + 1)) for row in range(batch_size)]
    inputs = (
        torch.ones((batch_size, length), dtype=torch.int64, device="cuda"),
        torch.stack([torch.arange(start, start + length, device="cuda") for start in starts]),
        cache,
        torch.tensor([start + count for start, count in zip(starts, counts)], dtype=torch.int32),
        torch.tensor(counts, dtype=torch.int32),
    )
    batch = torch.export.Dim("batch", min=1, max=MAX_BATCH_SIZE)
    token_shape = {0: batch}
    if kind == "prefill":
        token_shape[1] = torch.export.Dim("tokens", min=2, max=MAX_PREFILL)
    shapes = (
        token_shape,
        token_shape,
        tuple({0: batch} for _ in cache),
        {0: batch},
        {0: batch},
    )
    return inputs, shapes


def export_step(step, inputs, shapes):
    # Keep B=1 inside the symbolic range used by AOTI's input guards.
    with symbolic_config.patch(backed_size_oblivious=True):
        exported = torch.export.export(step, inputs, dynamic_shapes=shapes, strict=False)
    exported.validate()
    return exported


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
    # Text dumps expose mutation/shape behavior without another copy of the weights.
    (staging_directory / f"{kind}.graph.txt").write_text(exported.graph_module.code)
    print(f"Exported {kind}; compiling...", flush=True)

    def metadata_factory(_compiled_path):
        cache_shape = (MAX_BATCH_SIZE, *inputs[2][0].shape[1:])
        metadata = entry_metadata(model, kind, checkpoint, capacity, cache_shape)
        report_export(
            "qwen3",
            kind,
            strict=False,
            diagnostics=dict(
                graph_nodes=len(list(exported.graph.nodes)),
                export_and_compile_seconds=time.perf_counter() - start,
                model_adapter=dict(name="qwen3_flash_v1"),
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
    model = load_model(model_dir)
    step = FlashStep(model)
    manifest = read_json(model_dir / MODEL_MANIFEST_NAME)
    revision = manifest["revision"]
    checkpoint = dict(model_id=manifest["repo_id"], revision=revision, manifest=manifest)
    # Retain BF16 cast boundaries; fused arithmetic can still differ from eager.
    options = {"emulate_precision_casts": True}
    prepare_entry = partial(
        export_entry,
        model=model,
        step=step,
        checkpoint=checkpoint,
        capacity=capacity,
    )
    with torch.inference_mode(), symbolic_config.patch(backed_size_oblivious=True):
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
