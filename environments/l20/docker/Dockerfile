ARG XINFER_BASE_IMAGE
FROM ${XINFER_BASE_IMAGE} AS build

USER 0:0
SHELL ["/bin/bash", "-o", "pipefail", "-c"]

ENV PYTHONHOME="" \
    PYTHONPATH="" \
    LD_PRELOAD="" \
    PYTHONNOUSERSITE=1 \
    PIP_CONFIG_FILE=/dev/null \
    PIP_NO_INDEX=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    NVIDIA_VISIBLE_DEVICES=void \
    LD_LIBRARY_PATH=/opt/rh/gcc-toolset-11/root/usr/lib64:/opt/xinfer/native/lib \
    PATH=/opt/xinfer/native/bin:/usr/local/cuda/bin:/opt/rh/gcc-toolset-11/root/usr/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin

WORKDIR /opt/xinfer/offline
COPY native.lock qwen3.lock qwen35.lock SHA256SUMS ./
COPY payload/ ./payload/
RUN sha256sum -c SHA256SUMS \
    && mkdir -p /opt/xinfer/native /opt/xinfer/deps/nlohmann_json \
    && tar -xzf payload/python-runtime.tar.gz --strip-components=1 -C /opt/xinfer/native \
    && rm -f /opt/xinfer/native/lib/python3.12/EXTERNALLY-MANAGED \
    && rm -rf /opt/xinfer/native/lib/python3.12/site-packages \
    && mkdir -p /opt/xinfer/native/lib/python3.12/site-packages \
    && ln -sfn python3.12 /opt/xinfer/native/bin/python \
    && ln -sfn python3.12 /opt/xinfer/native/bin/python3 \
    && python -m ensurepip --upgrade --default-pip \
    && python -m pip install --no-index --find-links=payload/wheels -r native.lock \
    && python -m pip check \
    && python -m venv --system-site-packages /opt/xinfer/qwen3 \
    && /opt/xinfer/qwen3/bin/python -m pip install --no-index --find-links=payload/wheels -c native.lock -r qwen3.lock \
    && /opt/xinfer/qwen3/bin/python -m pip check \
    && python -m venv --system-site-packages /opt/xinfer/qwen35 \
    && /opt/xinfer/qwen35/bin/python -m pip install --no-index --find-links=payload/wheels -c native.lock -r qwen35.lock \
    && /opt/xinfer/qwen35/bin/python -m pip check \
    && tar -xJf payload/nlohmann-json.tar.xz --strip-components=1 -C /opt/xinfer/deps/nlohmann_json

# Build-time checks do not require a GPU. Device execution is checked after launch.
RUN python -c 'import torch, triton; assert torch.__version__ == "2.13.0+cu126"; assert torch.version.cuda == "12.6"; assert triton.__version__ == "3.7.1"; assert torch.arange(4).sum().item() == 6; print(torch.__version__, triton.__version__)' \
    && /opt/xinfer/qwen3/bin/python -c 'import importlib.metadata as m; import transformers; assert transformers.__version__ == "4.51.3"; assert m.version("tokenizers") == "0.21.4"; assert m.version("safetensors") == "0.5.3"; assert m.version("huggingface-hub") == "0.30.2"' \
    && /opt/xinfer/qwen35/bin/python -c 'import importlib.metadata as m; import importlib.util as u; import transformers; assert transformers.__version__ == "5.20.2"; assert m.version("tokenizers") == "0.23.3"; assert m.version("safetensors") == "0.8.0"; assert m.version("huggingface-hub") == "2.3.0"; assert u.find_spec("fla") is None; assert u.find_spec("causal_conv1d") is None' \
    && cmake --version \
    && ninja --version \
    && nvcc --version \
    && c++ --version \
    && ruff --version \
    && clang-format --version \
    && mkdir -p /opt/xinfer/recipe \
    && cp native.lock qwen3.lock qwen35.lock SHA256SUMS /opt/xinfer/recipe/

COPY Dockerfile .dockerignore README.md launch.sh /opt/xinfer/recipe/

FROM ${XINFER_BASE_IMAGE}

ARG XINFER_BASE_IMAGE

USER 0:0
SHELL ["/bin/bash", "-o", "pipefail", "-c"]

LABEL org.opencontainers.image.title="xinfer L20 testing environment" \
      org.opencontainers.image.description="Offline Python 3.12, Torch 2.13 cu126 and Triton 3.7.1 with isolated Qwen model dependencies" \
      org.opencontainers.image.base.name="${XINFER_BASE_IMAGE}"

COPY --from=build /opt/xinfer/native /opt/xinfer/native
COPY --from=build /opt/xinfer/qwen3 /opt/xinfer/qwen3
COPY --from=build /opt/xinfer/qwen35 /opt/xinfer/qwen35
COPY --from=build /opt/xinfer/deps /opt/xinfer/deps
COPY --from=build /opt/xinfer/recipe /opt/xinfer/recipe

ENV XINFER_CUDA_LIBS=/opt/xinfer/native/lib/python3.12/site-packages/nvidia/cublas/lib:/opt/xinfer/native/lib/python3.12/site-packages/nvidia/cuda_cupti/lib:/opt/xinfer/native/lib/python3.12/site-packages/nvidia/cuda_nvrtc/lib:/opt/xinfer/native/lib/python3.12/site-packages/nvidia/cuda_runtime/lib:/opt/xinfer/native/lib/python3.12/site-packages/nvidia/cudnn/lib:/opt/xinfer/native/lib/python3.12/site-packages/nvidia/cufft/lib:/opt/xinfer/native/lib/python3.12/site-packages/nvidia/cufile/lib:/opt/xinfer/native/lib/python3.12/site-packages/nvidia/curand/lib:/opt/xinfer/native/lib/python3.12/site-packages/nvidia/cusolver/lib:/opt/xinfer/native/lib/python3.12/site-packages/nvidia/cusparse/lib:/opt/xinfer/native/lib/python3.12/site-packages/nvidia/cusparselt/lib:/opt/xinfer/native/lib/python3.12/site-packages/nvidia/nccl/lib:/opt/xinfer/native/lib/python3.12/site-packages/nvidia/nvjitlink/lib:/opt/xinfer/native/lib/python3.12/site-packages/nvidia/nvshmem/lib:/opt/xinfer/native/lib/python3.12/site-packages/nvidia/nvtx/lib

ENV PYTHONHOME="" \
    PYTHONPATH="" \
    LD_PRELOAD="" \
    PYTHONNOUSERSITE=1 \
    PIP_CONFIG_FILE=/dev/null \
    PIP_NO_INDEX=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    NVIDIA_VISIBLE_DEVICES=void \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    CUDA_HOME=/usr/local/cuda \
    NATIVE_PY=/opt/xinfer/native/bin/python \
    Q3_PY=/opt/xinfer/qwen3/bin/python \
    Q35_PY=/opt/xinfer/qwen35/bin/python \
    CMAKE=/opt/xinfer/native/bin/cmake \
    FETCHCONTENT_SOURCE_DIR_NLOHMANN_JSON=/opt/xinfer/deps/nlohmann_json \
    HF_HOME=/cache/huggingface \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    HF_HUB_DISABLE_TELEMETRY=1 \
    TORCHINDUCTOR_CACHE_DIR=/cache/torchinductor \
    TRITON_CACHE_DIR=/cache/triton \
    CUDA_CACHE_PATH=/cache/cuda \
    XDG_CACHE_HOME=/cache/xdg \
    TMPDIR=/cache/tmp \
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    TORCH_CUDA_ARCH_LIST=8.9 \
    LD_LIBRARY_PATH=${XINFER_CUDA_LIBS}:/opt/rh/gcc-toolset-11/root/usr/lib64:/opt/xinfer/native/lib \
    LDFLAGS="-Wl,-rpath-link,${XINFER_CUDA_LIBS}" \
    PATH=/opt/xinfer/qwen35/bin:/opt/xinfer/native/bin:/usr/local/cuda/bin:/opt/rh/gcc-toolset-11/root/usr/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin

RUN mkdir -p /workspace/xinfer /models /cache/tmp /reports
WORKDIR /workspace/xinfer
ENTRYPOINT []
CMD ["sleep", "infinity"]
