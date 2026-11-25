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
from compiler.qwen3.model import MAX_BATCH_SIZE, MAX_PREFILL, ModelStep, allocate_cache, load_model
from compiler.qwen3.config import DEFAULT_MODEL


def export_step(step, kind, capacity):
    """Build tracing inputs and export the adapted model for one entry point."""
    batch_size = 2
    length = 1 if kind == "decode" else 16
    cache = allocate_cache(step.model.config, capacity, batch_size)
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
    # Keep B=1 inside the symbolic range used by AOTI's input guards.
    with symbolic_config.patch(backed_size_oblivious=True):
        exported = torch.export.export(step, inputs, dynamic_shapes=shapes, strict=False)
    exported.validate()
    return exported


def bundle_metadata(model, checkpoint, capacity):
    config = model.config
    cache_shape = [MAX_BATCH_SIZE, config.num_key_value_heads, capacity, config.head_dim]
    state = [
        state_spec(layer, name, cache_shape, "bfloat16", "inplace")
        for layer in range(config.num_hidden_layers)
        for name in ("key", "value")
    ]
    return make_metadata(
        model_type="qwen3",
        checkpoint=checkpoint,
        torch_version=torch.__version__,
        cxx11_abi=torch._C._GLIBCXX_USE_CXX11_ABI,
        cuda_compute_capability=list(torch.cuda.get_device_capability()),
        cache_length=capacity,
        max_prefill_tokens=MAX_PREFILL,
        vocab_size=config.vocab_size,
        eos_token_ids=model.generation_config.eos_token_id,
        state=state,
    )


def export_entry(kind, staging_directory, *, step, capacity):
    start = time.perf_counter()
    exported = export_step(step, kind, capacity)
    # Text dumps expose mutation/shape behavior without another copy of the weights.
    (staging_directory / f"{kind}.graph.txt").write_text(exported.graph_module.code)
    print(f"Exported {kind}; compiling...", flush=True)

    def on_compiled(compiled_path):
        report_export(
            "qwen3",
            kind,
            strict=False,
            package_bytes=compiled_path.stat().st_size,
            diagnostics=dict(
                graph_nodes=len(list(exported.graph.nodes)),
                export_and_compile_seconds=time.perf_counter() - start,
                model_adapter=dict(name="qwen3_flash_v1"),
            ),
        )

    return exported, on_compiled


def export_model(*, model_dir, output, max_context_length=None):
    """Load one adapted model and publish its paired shared-weight bundle."""
    model_dir, output = Path(model_dir), Path(output)
    capacity = prepare_export(model_dir, output, max_context_length, MAX_PREFILL)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model = load_model(model_dir)
    step = ModelStep(model)
    checkpoint = read_json(model_dir / MODEL_MANIFEST_NAME)
    # Retain BF16 cast boundaries; fused arithmetic can still differ from eager.
    options = {"emulate_precision_casts": True}
    prepare_entry = partial(export_entry, step=step, capacity=capacity)
    with torch.inference_mode(), symbolic_config.patch(backed_size_oblivious=True):
        compile_shared(
            output,
            prepare_entry,
            identity=dict(model="qwen3", checkpoint=checkpoint),
            inductor_configs=options,
            metadata=partial(bundle_metadata, model, checkpoint, capacity),
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
