"""AOT-compile Qwen3.5 text: dynamic prefill and decode with packaged Triton GDN."""

import argparse
import importlib.metadata
import json
import time
from pathlib import Path

import torch

from compiler.common.artifacts import MODEL_MANIFEST_NAME, read_json
from compiler.common.context import model_context_limit, resolve_context_length
from compiler.qwen35.config import ARTIFACT_DIR, MAX_PREFILL, TEXT_DIR
from compiler.common.passes.pack_linears import pack_linears
from compiler.common.weights import compile_shared
from compiler.common.attention import cache_extent
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
    return exported, packing


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=TEXT_DIR)
    parser.add_argument("--output", type=Path, default=ARTIFACT_DIR / "text")
    parser.add_argument(
        "--max-context-length",
        type=int,
        help="Compiled KV-cache capacity; defaults to model max_position_embeddings",
    )
    parser.add_argument(
        "--enable-inductor-optimization",
        action="store_true",
        help="Enable optional Inductor optimizations (currently combo kernels)",
    )
    args = parser.parse_args()
    if args.output.is_symlink() or (
        args.output.exists() and (not args.output.is_dir() or any(args.output.iterdir()))
    ):
        parser.error(f"Output must be a new or empty directory: {args.output}")
    try:
        context_config = read_json(args.model_dir / "config.json")
        capacity = resolve_context_length(context_config, args.max_context_length)
        ceiling = model_context_limit(context_config)
    except ValueError as error:
        parser.error(str(error))
    print(
        json.dumps(
            dict(
                event="context",
                cache_length=capacity,
                model_max_position_embeddings=ceiling,
                max_prefill_tokens=MAX_PREFILL,
            )
        ),
        flush=True,
    )
    configure_text()
    from torch.fx.experimental import _config as symbolic_config

    if symbolic_config.backed_size_oblivious:
        raise ValueError("Flash must export with backed_size_oblivious disabled")
    model = load_model(args.model_dir)
    install_delta(model)
    config = model.config
    checkpoint = json.loads((args.model_dir / MODEL_MANIFEST_NAME).read_text())
    options = {"emulate_precision_casts": True}
    if args.enable_inductor_optimization:
        options.update(combo_kernels=True, benchmark_combo_kernel=True)

    def export_entry(kind, staging_directory):
        dynamic = kind == "prefill"
        length = 32 if dynamic else 1
        start = time.perf_counter()
        exported, packing = export_step(model, kind, capacity)
        (staging_directory / f"{kind}.packing.json").write_text(
            json.dumps(packing, indent=2) + "\n"
        )
        print(
            f"Packed {len(packing['groups'])} groups: "
            f"{packing['linear_calls_before']} -> {packing['linear_calls_after']} linear calls",
            flush=True,
        )
        export_seconds = time.perf_counter() - start
        assert not exported.graph_signature.buffers_to_mutate
        targets = [node.target for node in exported.graph.nodes if node.op == "call_function"]
        full_layers = config.layer_types.count("full_attention")
        assert targets.count(torch.ops.xinfer.flash_attention.default) == full_layers
        assert targets.count(torch.ops.aten.index_copy_.default) == 2 * full_layers
        assert torch.ops.aten.index_copy.default not in targets
        assert torch.ops.aten.scaled_dot_product_attention.default not in targets
        triton_ops = config.layer_types.count("linear_attention")
        graph = delta_graph(exported, kind, triton_ops)
        (staging_directory / f"{kind}.graph.txt").write_text(
            exported.graph_module.print_readable(print_output=False) + "\n"
        )
        print(
            f"Exported {kind}: {len(list(exported.graph.nodes))} nodes; "
            f"DeltaNet branches {graph}; ranges {exported.range_constraints}",
            flush=True,
        )
        start = time.perf_counter()

        def metadata_factory(compiled_path):
            from torch._inductor.runtime.triton_helpers import get_backend_options

            diagnostics = dict(
                delta_graph=graph,
                range_constraints=str(exported.range_constraints),
                graph_nodes=len(list(exported.graph.nodes)),
                export_seconds=export_seconds,
                compile_seconds=time.perf_counter() - start,
            )
            metadata = make_metadata(
                model_type="qwen35",
                entry_point=kind,
                token_range=[2, MAX_PREFILL] if dynamic else [length, length],
                checkpoint=dict(
                    model_id=checkpoint["model"],
                    revision=checkpoint["revision"],
                    manifest=checkpoint,
                ),
                torch_version=torch.__version__,
                cxx11_abi=torch._C._GLIBCXX_USE_CXX11_ABI,
                cuda_compute_capability=torch.cuda.get_device_capability(),
                cache_length=capacity,
                vocab_size=config.vocab_size,
                eos_token_ids=model.generation_config.eos_token_id,
                state=state_specs(config, capacity),
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
            print(
                json.dumps(
                    dict(
                        event="export",
                        model_type="qwen35",
                        entry_point=kind,
                        package_bytes=compiled_path.stat().st_size,
                        build=dict(
                            strict_export=True,
                            torch_cuda_version=torch.version.cuda,
                            versions={
                                name: importlib.metadata.version(name)
                                for name in (
                                    "transformers",
                                    "triton",
                                    "safetensors",
                                    "huggingface-hub",
                                    "tokenizers",
                                )
                            },
                        ),
                        diagnostics=dict(
                            **diagnostics,
                            triton_delta_ops=triton_ops,
                            linear_packing=packing,
                            hidden_size=config.hidden_size,
                            layer_types=config.layer_types,
                            parameter_dtypes=sorted({str(p.dtype) for p in model.parameters()}),
                            rope_frequency_dtype=str(model.model.rotary_emb.inv_freq.dtype),
                            tied_embeddings=model.lm_head.weight is model.model.embed_tokens.weight,
                            enable_inductor_optimization=args.enable_inductor_optimization,
                        ),
                    )
                ),
                flush=True,
            )
            return metadata

        return exported, metadata_factory

    with torch.inference_mode():
        compile_shared(
            args.output,
            export_entry,
            identity=dict(model="qwen3.5", checkpoint=checkpoint),
            inductor_configs=options,
        )
    print(f"Saved prefill/decode bundle to {args.output}", flush=True)


if __name__ == "__main__":
    main()
