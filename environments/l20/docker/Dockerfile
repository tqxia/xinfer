FROM nvidia/cuda:12.6.3-devel-ubuntu24.04

USER 0:0
SHELL ["/bin/bash", "-o", "pipefail", "-c"]

LABEL org.opencontainers.image.title="xinfer L20 testing environment" \
      org.opencontainers.image.description="Python 3.12, PyTorch, Triton and native build tools for xinfer on L20"

RUN DEBIAN_FRONTEND=noninteractive apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        python3.12 \
        python3.12-venv \
        python3.12-dev \
        nlohmann-json3-dev \
        git \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY environments/requirements.txt /tmp/xinfer/environments/requirements.txt
COPY environments/l20/requirements.txt /tmp/xinfer/environments/l20/requirements.txt
COPY requirements-dev.txt /tmp/xinfer/requirements-dev.txt

ARG PIP_INDEX_URL
RUN python3.12 -m venv /opt/xinfer/venv \
    && /opt/xinfer/venv/bin/pip install --no-cache-dir \
        -r /tmp/xinfer/environments/l20/requirements.txt \
        -r /tmp/xinfer/requirements-dev.txt \
    && /opt/xinfer/venv/bin/pip check \
    && /opt/xinfer/venv/bin/python -c 'import torch, triton, transformers, kernels; print(torch.__version__, triton.__version__)' \
    && rm -rf /tmp/xinfer

ENV PY=/opt/xinfer/venv/bin/python \
    PATH="/opt/xinfer/venv/bin:${PATH}" \
    CUDA_HOME=/usr/local/cuda \
    TORCH_CUDA_ARCH_LIST=8.9 \
    PYTHONNOUSERSITE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    NVIDIA_VISIBLE_DEVICES=void \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    HF_HOME=/cache/huggingface \
    HF_HUB_DISABLE_TELEMETRY=1 \
    TORCHINDUCTOR_CACHE_DIR=/cache/torchinductor \
    TRITON_CACHE_DIR=/cache/triton \
    CUDA_CACHE_PATH=/cache/cuda \
    XDG_CACHE_HOME=/cache/xdg \
    TMPDIR=/cache/tmp \
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1

RUN mkdir -p /workspace/xinfer /models /cache/tmp /reports
WORKDIR /workspace/xinfer
ENTRYPOINT []
CMD ["sleep", "infinity"]
