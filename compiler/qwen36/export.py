"""AOT-compile the complete Qwen3.6 FP8 text model with shared prefill/decode weights."""

from functools import partial
import json
from pathlib import Path
import time

import torch

from compiler.common.artifacts import MODEL_MANIFEST_NAME, read_json, require
from compiler.common.attention import (
    cache_extent,
    load_attention_ops,
    attention_library,
    attention_c_shims,
)
from compiler.common.context import resolve_context_length
from compiler.common.export import (
    ExportInputError,
    check_symbolic_config,
    export_parser,
    prepare_export,
    report_export,
)
from compiler.common.kernels.fp8 import KERNEL_REPO, KERNEL_REVISION
from compiler.common.metadata import make_metadata
from compiler.patches.triton_access import ANALYZER_SHA256, memoize_triton_reads
from compiler.common.weights import compile_shared
from compiler.qwen35.adapters.delta import install_delta
from compiler.qwen35.config import DELTA_OPERATORS
from compiler.qwen35.kernels.chunked import CHUNK_SIZE, TRANSFORM_TILE
from compiler.qwen35.kernels.recurrent import SHORT_THRESHOLD
from compiler.qwen36.adapters import install_fp8
from compiler.qwen36.config import ARTIFACT_DIR, MAX_PREFILL, TEXT_DIR, configure
from compiler.qwen36.model import TextStep, load_model, state_specs, zero_state


def check_model(model):
    """A production bundle always contains the complete pinned text architecture."""
    config = model.config
    expected = dict(
        num_hidden_layers=40,
        hidden_size=2048,
        num_attention_heads=16,
        num_key_value_heads=2,
        head_dim=256,
        linear_num_key_heads=16,
        linear_num_value_heads=32,
        linear_key_head_dim=128,
        linear_value_head_dim=128,
        linear_conv_kernel_dim=4,
        num_experts=256,
        num_experts_per_tok=8,
        moe_intermediate_size=512,
        shared_expert_intermediate_size=512,
        vocab_size=248320,
    )
    require(
        all(getattr(config, name) == value for name, value in expected.items())
        and len(model.model.layers) == 40
        and config.layer_types == (["linear_attention"] * 3 + ["full_attention"]) * 10,
        "Qwen3.6 export requires the complete 40-layer text model",
    )
    require(
        config.tie_word_embeddings is False
        and model.lm_head.weight.data_ptr() != model.model.embed_tokens.weight.data_ptr()
        and model.generation_config.eos_token_id == [248046, 248044],
        "Qwen3.6 export requires the untied head and both checkpoint EOS IDs",
    )
    require(
        {value.dtype for value in model.parameters()} == {torch.float8_e4m3fn, torch.bfloat16},
        "Qwen3.6 export must preserve mixed FP8/BF16 checkpoint parameters",
    )


def export_step(model, kind, capacity=None):
    """Strict dynamic prefill or one-token decode, without BF16 linear packing."""
    require(kind in ("prefill", "decode"), "Unknown export entry point")
    capacity = resolve_context_length(model.config, capacity)
    dynamic = kind == "prefill"
    length = 32 if dynamic else 1
    device = model.model.embed_tokens.weight.device
    inputs = (
        torch.randint(0, model.config.vocab_size, (1, length), device=device),
        torch.arange(length, device=device),
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
    return exported


def model_specific():
    return dict(
        attention_backend="flash_attention",
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
        fp8_backend="triton_op",
        fp8_format="e4m3fn",
        fp8_block_size=[128, 128],
        fp8_scale_dtype="bfloat16",
        fp8_kernel=dict(repo=KERNEL_REPO, revision=KERNEL_REVISION),
        linear_packing=dict(enabled=False),
    )


def entry_metadata(model, kind, checkpoint, capacity):
    from torch._inductor.runtime.triton_helpers import get_backend_options

    return make_metadata(
        model_type="qwen36",
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
        model_specific=model_specific(),
    )


def model_diagnostics(model, adapters, enable_inductor_optimization):
    return dict(
        fp8_adapters=adapters,
        checkpoint_storage=model.checkpoint_storage,
        parameter_dtypes=sorted({str(p.dtype) for p in model.parameters()}),
        rope_frequency_dtype=str(model.model.rotary_emb.inv_freq.dtype),
        tied_embeddings=False,
        linear_packing=False,
        enable_inductor_optimization=enable_inductor_optimization,
        hf_numerical_acceptance="unresolved; export does not establish HF accuracy",
    )


def export_entry(
    kind, staging_directory, *, model, checkpoint, capacity, adapters, enable_inductor_optimization
):
    start = time.perf_counter()
    exported = export_step(model, kind, capacity)
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

    def metadata_factory(compiled_path):
        metadata = entry_metadata(model, kind, checkpoint, capacity)
        report_export(
            "qwen36",
            kind,
            strict=True,
            extra_dependencies=("kernels",),
            package_bytes=compiled_path.stat().st_size,
            diagnostics=dict(
                range_constraints=str(exported.range_constraints),
                graph_nodes=len(list(exported.graph.nodes)),
                export_seconds=export_seconds,
                compile_seconds=time.perf_counter() - compile_start,
                **model_diagnostics(model, adapters, enable_inductor_optimization),
            ),
        )
        return metadata

    return exported, metadata_factory


def export_model(*, model_dir, output, max_context_length=None, enable_inductor_optimization=False):
    """Load the complete FP8 model and publish its paired shared-weight bundle."""
    model_dir, output = Path(model_dir), Path(output)
    capacity = prepare_export(model_dir, output, max_context_length, MAX_PREFILL)
    configure()
    check_symbolic_config()
    model = load_model(model_dir)
    check_model(model)
    adapters = install_fp8(model)
    require(adapters == dict(linear=250, experts=40), "Incomplete FP8 model adaptation")
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
        adapters=adapters,
        enable_inductor_optimization=enable_inductor_optimization,
    )
    with torch.inference_mode(), memoize_triton_reads():
        print(
            json.dumps(
                dict(
                    event="compiler_compatibility",
                    triton_read_dag_memoized=True,
                    analyzer_sha256=ANALYZER_SHA256,
                )
            ),
            flush=True,
        )
        load_attention_ops(model="qwen36")
        compile_shared(
            output,
            prepare_entry,
            identity=dict(model="qwen3.6", checkpoint=checkpoint),
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
