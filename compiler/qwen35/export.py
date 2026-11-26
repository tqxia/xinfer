"""AOT-compile Qwen3.5 text: dynamic prefill and decode with bundled native ops."""

from functools import partial
import json
from pathlib import Path

import torch
from torch.fx.experimental import _config as symbolic_config

from compiler.common.artifacts import MODEL_MANIFEST_NAME, read_json
from compiler.common.context import resolve_context_length
from compiler.common.passes.pack_linears import pack_linears
from compiler.common.passes.projection_gemv import replace_projection_gemv
from compiler.qwen35.config import ARTIFACT_DIR, MAX_BATCH_SIZE, MAX_PREFILL, TEXT_DIR, configure
from compiler.common.compile import compile_shared
from compiler.patches.torch_2_13.export_cleanup import cleanup_export_globals
from compiler.common.kernels.flash_attention.attention import (
    attention_library,
    attention_c_shims,
    load_attention_ops,
)
from compiler.common.export import (
    ExportInputError,
    export_parser,
    prepare_export,
)
from compiler.common.metadata import make_metadata
from compiler.common.kernels.gated_delta_net.deltanet import (
    deltanet_c_shims,
    deltanet_library,
    load_deltanet_ops,
)
from compiler.qwen35.model import ModelStep, load_model, state_specs, zero_state
from compiler.common.adapters.gated_delta_net import install_delta


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
    shapes = (
        token_shape,
        token_shape,
        tuple({0: batch} for _ in inputs[2]),
        {0: batch},
        {0: batch},
    )
    # Keep B=1 inside the symbolic range used by AOTI's input guards.
    with symbolic_config.patch(backed_size_oblivious=True), cleanup_export_globals():
        exported = torch.export.export(ModelStep(model), inputs, strict=True, dynamic_shapes=shapes)
    exported.validate()
    packing = pack_linears(exported)
    if kind == "decode":
        projections = replace_projection_gemv(exported)
        print(json.dumps(dict(event="decoder_projection_gemv", **projections)), flush=True)
    return exported, packing


def bundle_metadata(model, checkpoint, capacity, model_type="qwen35"):
    return make_metadata(
        model_type=model_type,
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


def export_entry(kind, staging_directory, *, model, capacity, step_exporter):
    exported, packing = step_exporter(model, kind, capacity)
    (staging_directory / f"{kind}.packing.json").write_text(json.dumps(packing, indent=2) + "\n")
    print(
        f"Packed {len(packing['groups'])} groups: "
        f"{packing['linear_calls_before']} -> {packing['linear_calls_after']} linear calls",
        flush=True,
    )
    (staging_directory / f"{kind}.graph.txt").write_text(
        exported.graph_module.print_readable(print_output=False) + "\n"
    )
    print(
        f"Exported {kind}: {len(list(exported.graph.nodes))} nodes; "
        f"ranges {exported.range_constraints}",
        flush=True,
    )
    return exported


def export_model(
    *,
    model_dir,
    output,
    max_context_length=None,
    enable_inductor_optimization=False,
    model_type="qwen35",
    model_loader=None,
    step_exporter=None,
):
    """Load one adapted model and publish its paired shared-weight bundle."""
    model_dir, output = Path(model_dir), Path(output)
    capacity = prepare_export(model_dir, output, max_context_length, MAX_PREFILL)
    configure()
    model = (load_model if model_loader is None else model_loader)(model_dir)
    install_delta(model)
    checkpoint = read_json(model_dir / MODEL_MANIFEST_NAME)
    options = {"emulate_precision_casts": True}
    if enable_inductor_optimization:
        options.update(combo_kernels=True, benchmark_combo_kernel=True)
    prepare_entry = partial(
        export_entry,
        model=model,
        capacity=capacity,
        step_exporter=export_step if step_exporter is None else step_exporter,
    )
    with torch.inference_mode(), symbolic_config.patch(backed_size_oblivious=True):
        load_attention_ops()
        load_deltanet_ops()
        compile_shared(
            output,
            prepare_entry,
            identity=dict(
                model={"qwen35": "qwen3.5", "qwen38": "qwen3.8"}[model_type], checkpoint=checkpoint
            ),
            inductor_configs=options,
            metadata=partial(bundle_metadata, model, checkpoint, capacity, model_type),
            native_libraries=[attention_library(), deltanet_library()],
            custom_op_shims={**attention_c_shims(), **deltanet_c_shims()},
        )
    print(f"Saved prefill/decode bundle to {output}", flush=True)


def main(
    argv=None,
    *,
    description=__doc__,
    model_dir=TEXT_DIR,
    output=ARTIFACT_DIR / "text",
    model_type="qwen35",
    model_loader=None,
    step_exporter=None,
):
    parser = export_parser(description, model_dir=model_dir, output=output)
    parser.add_argument(
        "--enable-inductor-optimization",
        action="store_true",
        help="Enable optional Inductor optimizations (currently combo kernels)",
    )
    args = parser.parse_args(argv)
    try:
        export_model(
            **vars(args),
            model_type=model_type,
            model_loader=model_loader,
            step_exporter=step_exporter,
        )
    except ExportInputError as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
