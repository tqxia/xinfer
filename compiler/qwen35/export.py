"""AOT-compile Qwen3.5 text: dynamic prefill and decode with packaged Triton GDN."""

import argparse
import importlib.metadata
import inspect
import json
import os
import time
from pathlib import Path

import torch
from transformers.models.qwen3_5 import modeling_qwen3_5

from compiler.qwen35.config import ARTIFACT_DIR, CACHE_LENGTH, TEXT_DIR
from compiler.common.artifacts import sha256, ROOT
from compiler.common.passes.pack_linears import pack_linears
from compiler.qwen35.model import TextStep, configure_text, load_model, state_specs, zero_state
from compiler.qwen35.adapters.delta import install_delta
from compiler.qwen35.kernels.chunked import CHUNK_SIZE


def export_step(model, kind):
    """Create the graph used by compilation and the packing-only eager control."""
    dynamic = kind == "prefill"
    length = 32 if dynamic else 1
    torch.manual_seed(35)
    inputs = (torch.randint(0, model.config.vocab_size, (1, length), device="cuda"),
              torch.arange(length, device="cuda"), zero_state(model.config))
    shapes = None
    if dynamic:
        tokens = torch.export.Dim("tokens", min=2, max=512)
        shapes = ({1: tokens}, {0: tokens}, [None] * len(inputs[2]))
    exported = torch.export.export(TextStep(model), inputs, strict=True, dynamic_shapes=shapes)
    exported.validate()
    packing = pack_linears(exported)
    return exported, packing


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("kind", choices=("prefill", "decode"))
    parser.add_argument("--model-dir", type=Path, default=TEXT_DIR)
    parser.add_argument("--output", type=Path, default=ARTIFACT_DIR / "text")
    parser.add_argument("--enable-inductor-optimization", action="store_true",
                        help="Enable optional Inductor optimizations (currently combo kernels)")
    parser.add_argument("--eager-numerics", action="store_true",
                        help="Keep native mean reductions and disable FP contraction")
    args = parser.parse_args()
    configure_text()
    args.output.mkdir(parents=True, exist_ok=True)
    package = args.output / f"{args.kind}.pt2"
    if package.exists():
        parser.error(f"Refusing to overwrite {package}")
    if args.eager_numerics:
        os.environ["TRITON_DEFAULT_FP_FUSION"] = "0"
        from torch._inductor.decomposition import decompositions
        from torch._inductor.lowering import make_fallback
        decompositions.pop(torch.ops.aten.mean.dim, None)
        make_fallback(torch.ops.aten.mean.dim, warn=False)
    model = load_model(args.model_dir)
    install_delta(model)
    config = model.config
    dynamic = args.kind == "prefill"
    length = 32 if dynamic else 1
    with torch.inference_mode():
        start = time.perf_counter()
        exported, packing = export_step(model, args.kind)
        (args.output / f"{args.kind}.packing.json").write_text(json.dumps(packing, indent=2) + "\n")
        print(f"Packed {len(packing['groups'])} groups: "
              f"{packing['linear_calls_before']} -> {packing['linear_calls_after']} linear calls", flush=True)
        export_seconds = time.perf_counter() - start
        assert not exported.graph_signature.buffers_to_mutate
        assert not exported.graph_signature.user_inputs_to_mutate
        delta_operator = "xinfer.delta_fused.default"
        triton_ops = sum(str(n.target) == delta_operator for n in exported.graph.nodes)
        assert triton_ops == config.layer_types.count("linear_attention")
        (args.output / f"{args.kind}.graph.txt").write_text(exported.graph_module.code)
        print(f"Exported {args.kind}: {len(list(exported.graph.nodes))} nodes; "
              f"{triton_ops} Triton DeltaNet ops; ranges {exported.range_constraints}", flush=True)
        options = {"emulate_precision_casts": True}
        if args.enable_inductor_optimization:
            options.update(combo_kernels=True, benchmark_combo_kernel=True)
        start = time.perf_counter()
        torch._inductor.aoti_compile_and_package(exported, package_path=str(package), inductor_configs=options)
        compile_seconds = time.perf_counter() - start
    metadata = dict(
        kind=args.kind, tokens=None if dynamic else length, token_range=[2, 512] if dynamic else [length, length],
        delta_backend="triton_op", delta_preparation="triton", delta_operator=delta_operator,
        delta_algorithm=dict(prefill="chunked_fp32", decode="fused_recurrent"),
        delta_chunk_size=CHUNK_SIZE, delta_dot_precision="ieee",
        linear_packing=dict(enabled=True, version=1,
                            groups=[{key: group[key] for key in
                                     ("packed_parameter", "source_parameters", "output_widths")}
                                    for group in packing["groups"]]),
        range_constraints=str(exported.range_constraints), triton_delta_ops=triton_ops,
        cache_length=CACHE_LENGTH, vocabulary=config.vocab_size,
        hidden_size=config.hidden_size, layer_types=config.layer_types,
        checkpoint=json.loads((args.model_dir / "manifest.json").read_text()),
        state=state_specs(config),
        input_order=["input_ids", "cache_position"] + [s["name"] for s in state_specs(config)],
        output_order=["last_logits"] + [s["name"] for s in state_specs(config)],
        logits=dict(shape=[1, config.vocab_size], dtype="bfloat16"), input_mutation=False,
        generation_config=model.generation_config.to_dict(),
        torch=torch.__version__, cuda=torch.version.cuda, cxx11_abi=torch._C._GLIBCXX_USE_CXX11_ABI,
        sm=torch.cuda.get_device_capability(), tf32=False, attention_backend="efficient_sdpa",
        versions={name: importlib.metadata.version(name) for name in
                  ("transformers", "triton", "safetensors", "huggingface-hub", "tokenizers")},
        hf_source_sha256=sha256(inspect.getfile(modeling_qwen3_5)),
        source_sha256={name: sha256(ROOT / name) for name in (
            'compiler/qwen35/config.py', 'compiler/qwen35/model.py', 'compiler/qwen35/export.py',
            'compiler/common/artifacts.py', 'compiler/qwen35/adapters/delta.py',
            'compiler/qwen35/kernels/recurrent.py', 'compiler/qwen35/kernels/chunked.py',
            'compiler/common/passes/pack_linears.py')},
        parameter_dtypes=sorted({str(p.dtype) for p in model.parameters()}),
        rope_frequency_dtype=str(model.model.rotary_emb.inv_freq.dtype),
        tied_embeddings=model.lm_head.weight is model.model.embed_tokens.weight,
        strict_export=True, graph_nodes=len(list(exported.graph.nodes)), inductor_configs=options,
        eager_numerics=args.eager_numerics,
        enable_inductor_optimization=args.enable_inductor_optimization,
        export_seconds=export_seconds, compile_seconds=compile_seconds,
        package_bytes=package.stat().st_size, package_sha256=sha256(package),
    )
    (args.output / f"{args.kind}.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps({k: metadata[k] for k in ("kind", "graph_nodes", "export_seconds", "compile_seconds", "package_bytes")}), flush=True)


if __name__ == "__main__":
    main()
