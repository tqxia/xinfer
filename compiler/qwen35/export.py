"""AOT-compile Qwen3.5 text: dynamic prefill and decode with packaged Triton GDN."""

from functools import partial
import json
import time
from pathlib import Path

import torch
from torch.fx.experimental import _config as symbolic_config

from compiler.common.artifacts import MODEL_MANIFEST_NAME, read_json
from compiler.common.context import resolve_context_length
from compiler.common.passes.pack_linears import pack_linears
from compiler.common.passes.projection_gemv import replace_projection_gemv
from compiler.qwen35.config import ARTIFACT_DIR, MAX_BATCH_SIZE, MAX_PREFILL, TEXT_DIR, configure
from compiler.common.compile import compile_shared
from compiler.common.kernels.flash_attention.attention import (
    attention_library,
    attention_c_shims,
    load_attention_ops,
)
from compiler.common.export import (
    ExportInputError,
    export_parser,
    prepare_export,
    report_export,
)
from compiler.common.metadata import make_metadata
from compiler.qwen35.model import ModelStep, load_model, state_specs, zero_state
from compiler.qwen35.adapters.delta import install_delta
from compiler.qwen35.kernels.chunked import CHUNK_SIZE, TRANSFORM_TILE
from compiler.qwen35.kernels.recurrent import SHORT_THRESHOLD


DELTA_OPERATORS = dict(
    short="xinfer.delta_cond_short.default", long="xinfer.delta_cond_long.default"
)


def export_step(model, kind, capacity=None):
    """Create the packed graph used by compilation and eager validation."""
    capacity = resolve_context_length(model.config, capacity)
    dynamic = kind == "prefill"
    length = 32 if dynamic else 1
    batch_size = 2
    torch.manual_seed(35)
    inputs = (
        torch.randint(0, model.config.vocab_size, (batch_size, length), device="cuda"),
        torch.arange(length, device="cuda").expand(batch_size, -1).contiguous(),
        zero_state(model.config, capacity, batch_size=batch_size),
        torch.full((batch_size,), length, dtype=torch.int32),
        torch.full((batch_size,), length, dtype=torch.int32),
    )
    batch = torch.export.Dim("batch", min=1, max=MAX_BATCH_SIZE)
    token_shape = {0: batch}
    if dynamic:
        token_shape[1] = torch.export.Dim("tokens", min=2, max=MAX_PREFILL)
    shapes = (token_shape, token_shape, [{0: batch}] * len(inputs[2]), {0: batch}, {0: batch})
    # Retain the singleton batch in AOTI guards, including the torch.cond branches.
    with symbolic_config.patch(backed_size_oblivious=True):
        exported = torch.export.export(ModelStep(model), inputs, strict=True, dynamic_shapes=shapes)
    exported.validate()
    packing = pack_linears(exported)
    if kind == "decode":
        projections = replace_projection_gemv(exported)
        print(json.dumps(dict(event="decoder_projection_gemv", **projections)), flush=True)
    return exported, packing


def bundle_metadata(model, checkpoint, capacity):
    return make_metadata(
        model_type="qwen35",
        checkpoint=checkpoint,
        torch_version=torch.__version__,
        cxx11_abi=torch._C._GLIBCXX_USE_CXX11_ABI,
        cuda_compute_capability=torch.cuda.get_device_capability(),
        cache_length=capacity,
        max_prefill_tokens=MAX_PREFILL,
        vocab_size=model.config.vocab_size,
        eos_token_ids=model.generation_config.eos_token_id,
        state=state_specs(model.config, capacity, batch_size=MAX_BATCH_SIZE),
    )


def export_entry(kind, staging_directory, *, model, capacity):
    start = time.perf_counter()
    exported, packing = export_step(model, kind, capacity)
    (staging_directory / f"{kind}.packing.json").write_text(json.dumps(packing, indent=2) + "\n")
    print(
        f"Packed {len(packing['groups'])} groups: "
        f"{packing['linear_calls_before']} -> {packing['linear_calls_after']} linear calls",
        flush=True,
    )
    export_seconds = time.perf_counter() - start
    (staging_directory / f"{kind}.graph.txt").write_text(
        exported.graph_module.print_readable(print_output=False) + "\n"
    )
    print(
        f"Exported {kind}: {len(list(exported.graph.nodes))} nodes; "
        f"ranges {exported.range_constraints}",
        flush=True,
    )
    compile_start = time.perf_counter()

    def on_compiled(compiled_path):
        diagnostics = dict(
            range_constraints=str(exported.range_constraints),
            graph_nodes=len(list(exported.graph.nodes)),
            export_seconds=export_seconds,
            compile_seconds=time.perf_counter() - compile_start,
            delta_backend="triton_op",
            delta_preparation="triton",
            delta_operator=DELTA_OPERATORS,
            delta_algorithm=dict(
                prefill="shape_cond",
                short="fused_recurrent",
                long="chunked_fp32",
                decode="fused_recurrent",
            ),
            delta_short_threshold=SHORT_THRESHOLD,
            delta_chunk_size=CHUNK_SIZE,
            delta_transform_tile=TRANSFORM_TILE,
            delta_dot_precision="ieee",
            linear_packing=dict(enabled=True, version=1),
        )
        report_export(
            "qwen35",
            kind,
            strict=True,
            package_bytes=compiled_path.stat().st_size,
            diagnostics=diagnostics,
        )

    return exported, on_compiled


def export_model(*, model_dir, output, max_context_length=None, enable_inductor_optimization=False):
    """Load one adapted model and publish its paired shared-weight bundle."""
    model_dir, output = Path(model_dir), Path(output)
    capacity = prepare_export(model_dir, output, max_context_length, MAX_PREFILL)
    configure()
    model = load_model(model_dir)
    install_delta(model)
    checkpoint = read_json(model_dir / MODEL_MANIFEST_NAME)
    options = {"emulate_precision_casts": True}
    if enable_inductor_optimization:
        options.update(combo_kernels=True, benchmark_combo_kernel=True)
    prepare_entry = partial(export_entry, model=model, capacity=capacity)
    with torch.inference_mode(), symbolic_config.patch(backed_size_oblivious=True):
        load_attention_ops(model="qwen35")
        compile_shared(
            output,
            prepare_entry,
            identity=dict(model="qwen3.5", checkpoint=checkpoint),
            inductor_configs=options,
            metadata=partial(bundle_metadata, model, checkpoint, capacity),
            native_libraries=[attention_library()],
            custom_op_shims=attention_c_shims(),
        )
    print(f"Saved prefill/decode bundle to {output}", flush=True)


def main(argv=None):
    parser = export_parser(__doc__, model_dir=TEXT_DIR, output=ARTIFACT_DIR / "text")
    parser.add_argument(
        "--enable-inductor-optimization",
        action="store_true",
        help="Enable optional Inductor optimizations (currently combo kernels)",
    )
    args = parser.parse_args(argv)
    try:
        export_model(**vars(args))
    except ExportInputError as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
