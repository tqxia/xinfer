"""AOT-compile Qwen3.5 text: dynamic prefill and decode with packaged Triton GDN."""

from functools import partial
import json
import time
from pathlib import Path

import torch

from compiler.common.artifacts import MODEL_MANIFEST_NAME, read_json
from compiler.common.context import resolve_context_length
from compiler.common.passes.pack_linears import pack_linears
from compiler.common.passes.projection_gemv import replace_projection_gemv
from compiler.qwen35.config import ARTIFACT_DIR, MAX_PREFILL, TEXT_DIR
from compiler.common.weights import compile_shared
from compiler.common.attention import (
    cache_extent,
    attention_library,
    attention_c_shims,
    load_attention_ops,
)
from compiler.common.export import (
    ExportInputError,
    check_symbolic_config,
    export_parser,
    prepare_export,
    report_export,
)
from compiler.common.metadata import make_metadata
from compiler.qwen35.model import TextStep, configure_text, load_model, state_specs, zero_state
from compiler.qwen35.adapters.delta import install_delta
from compiler.qwen35.kernels.chunked import CHUNK_SIZE, TRANSFORM_TILE
from compiler.qwen35.kernels.recurrent import SHORT_THRESHOLD


DELTA_OPERATORS = dict(
    short="xinfer.delta_cond_short.default", long="xinfer.delta_cond_long.default"
)


def delta_graph(exported, kind, layers):
    """Count branch calls, including repeated references to a shared subgraph."""
    counts = dict(cond=0, short=0, long=0)

    def visit(module):
        for node in module.graph.nodes:
            if node.target == torch.ops.higher_order.cond:
                predicate = node.args[0]
                assert isinstance(predicate, torch.fx.Node)
                assert isinstance(predicate.meta.get("val"), torch.SymBool), (
                    "Expected symbolic DeltaNet dispatch"
                )
                assert str(predicate.meta["val"]) not in ("True", "False"), (
                    "Specialized DeltaNet dispatch"
                )
                counts["cond"] += 1
                for branch in node.args[1:3]:
                    visit(module.get_submodule(branch.target))
            else:
                for branch, operator in DELTA_OPERATORS.items():
                    counts[branch] += str(node.target) == operator

    visit(exported.graph_module)
    dynamic = kind == "prefill"
    top_conditions = sum(n.target == torch.ops.higher_order.cond for n in exported.graph.nodes)
    assert top_conditions == (layers if dynamic else 0), "Unexpected top-level DeltaNet conditions"
    assert counts == dict(
        cond=layers if dynamic else 0, short=layers, long=layers if dynamic else 0
    ), counts
    return counts


def export_step(model, kind, capacity=None):
    """Create the packed graph used by compilation and eager validation."""
    capacity = resolve_context_length(model.config, capacity)
    dynamic = kind == "prefill"
    length = 32 if dynamic else 1
    torch.manual_seed(35)
    inputs = (
        torch.randint(0, model.config.vocab_size, (1, length), device="cuda"),
        torch.arange(length, device="cuda"),
        zero_state(model.config, capacity),
        cache_extent(length),
    )
    extent = torch.export.Dim("kv_extent", min=2, max=capacity + 1)
    shapes = ({}, {}, [None] * len(inputs[2]), {0: extent})
    if dynamic:
        tokens = torch.export.Dim("tokens", min=2, max=MAX_PREFILL)
        shapes = ({1: tokens}, {0: tokens}, [None] * len(inputs[2]), {0: extent})
    exported = torch.export.export(TextStep(model), inputs, strict=True, dynamic_shapes=shapes)
    exported.validate()
    packing = pack_linears(exported)
    if kind == "decode":
        projections = replace_projection_gemv(exported)
        print(json.dumps(dict(event="decoder_projection_gemv", **projections)), flush=True)
    return exported, packing


def check_graph(exported, kind, config):
    assert not exported.graph_signature.buffers_to_mutate
    targets = [node.target for node in exported.graph.nodes if node.op == "call_function"]
    full_layers = config.layer_types.count("full_attention")
    assert targets.count(torch.ops.xinfer.flash_attention.default) == full_layers
    assert targets.count(torch.ops.aten.index_copy_.default) == 2 * full_layers
    assert torch.ops.aten.index_copy.default not in targets
    assert torch.ops.aten.scaled_dot_product_attention.default not in targets
    return delta_graph(exported, kind, config.layer_types.count("linear_attention"))


def entry_metadata(model, kind, checkpoint, capacity):
    from torch._inductor.runtime.triton_helpers import get_backend_options

    return make_metadata(
        model_type="qwen35",
        entry_point=kind,
        token_range=[2, MAX_PREFILL] if kind == "prefill" else [1, 1],
        checkpoint=dict(
            model_id=checkpoint["model"],
            revision=checkpoint["revision"],
            manifest=checkpoint,
        ),
        torch_version=torch.__version__,
        cxx11_abi=torch._C._GLIBCXX_USE_CXX11_ABI,
        cuda_compute_capability=torch.cuda.get_device_capability(),
        cache_length=capacity,
        vocab_size=model.config.vocab_size,
        eos_token_ids=model.generation_config.eos_token_id,
        state=state_specs(model.config, capacity),
        triton_enable_fp_fusion=get_backend_options()["enable_fp_fusion"],
        model_specific=dict(
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
        ),
    )


def model_diagnostics(model, packing, enable_inductor_optimization):
    return dict(
        triton_delta_ops=model.config.layer_types.count("linear_attention"),
        linear_packing=packing,
        hidden_size=model.config.hidden_size,
        layer_types=model.config.layer_types,
        parameter_dtypes=sorted({str(p.dtype) for p in model.parameters()}),
        rope_frequency_dtype=str(model.model.rotary_emb.inv_freq.dtype),
        tied_embeddings=model.lm_head.weight is model.model.embed_tokens.weight,
        enable_inductor_optimization=enable_inductor_optimization,
    )


def export_entry(
    kind, staging_directory, *, model, checkpoint, capacity, enable_inductor_optimization
):
    start = time.perf_counter()
    exported, packing = export_step(model, kind, capacity)
    (staging_directory / f"{kind}.packing.json").write_text(json.dumps(packing, indent=2) + "\n")
    print(
        f"Packed {len(packing['groups'])} groups: "
        f"{packing['linear_calls_before']} -> {packing['linear_calls_after']} linear calls",
        flush=True,
    )
    export_seconds = time.perf_counter() - start
    graph = check_graph(exported, kind, model.config)
    (staging_directory / f"{kind}.graph.txt").write_text(
        exported.graph_module.print_readable(print_output=False) + "\n"
    )
    print(
        f"Exported {kind}: {len(list(exported.graph.nodes))} nodes; "
        f"DeltaNet branches {graph}; ranges {exported.range_constraints}",
        flush=True,
    )
    compile_start = time.perf_counter()

    def metadata_factory(compiled_path):
        diagnostics = dict(
            delta_graph=graph,
            range_constraints=str(exported.range_constraints),
            graph_nodes=len(list(exported.graph.nodes)),
            export_seconds=export_seconds,
            compile_seconds=time.perf_counter() - compile_start,
        )
        metadata = entry_metadata(model, kind, checkpoint, capacity)
        report_export(
            "qwen35",
            kind,
            strict=True,
            package_bytes=compiled_path.stat().st_size,
            diagnostics=dict(
                **diagnostics,
                **model_diagnostics(model, packing, enable_inductor_optimization),
            ),
        )
        return metadata

    return exported, metadata_factory


def export_model(*, model_dir, output, max_context_length=None, enable_inductor_optimization=False):
    """Load one adapted model and publish its paired shared-weight bundle."""
    model_dir, output = Path(model_dir), Path(output)
    capacity = prepare_export(model_dir, output, max_context_length, MAX_PREFILL)
    configure_text()
    check_symbolic_config()
    model = load_model(model_dir)
    install_delta(model)
    checkpoint = read_json(model_dir / MODEL_MANIFEST_NAME)
    options = {"emulate_precision_casts": True}
    if enable_inductor_optimization:
        options.update(combo_kernels=True, benchmark_combo_kernel=True)
    prepare_entry = partial(
        export_entry,
        model=model,
        checkpoint=checkpoint,
        capacity=capacity,
        enable_inductor_optimization=enable_inductor_optimization,
    )
    with torch.inference_mode():
        load_attention_ops(model="qwen35")
        compile_shared(
            output,
            prepare_entry,
            identity=dict(model="qwen3.5", checkpoint=checkpoint),
            inductor_configs=options,
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
