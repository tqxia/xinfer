# PRO 5000 Docker testing environment

Builds from `nvcr.io/nvidia/cuda:13.0.1-devel-ubuntu24.04` (CUDA 13.0.1, GCC
13.2, `ncu`) with Python 3.12, a minimal Rust toolchain (`/opt/rust`), and one
venv at `/opt/xinfer/venv` (`$PY`, first on `PATH`) installed directly from
[environments/pro5000/requirements.txt](../requirements.txt) and
[requirements-dev.txt](../../../requirements-dev.txt). Source, checkpoints, caches
and reports live in host bind mounts.

## Build and launch

Build from the repository root (`.dockerignore` excludes `.git`, `work`, `build`
and `artifacts`):

```bash
docker build -f environments/pro5000/docker/Dockerfile   -t xinfer/pro5000:torch2.13-cu130 .
```

Pass `--build-arg PIP_INDEX_URL=<index-url>` for a PyPI mirror and standard
`--build-arg http_proxy=...` / `https_proxy=...` if behind a proxy (Torch wheels
still pull from `https://download.pytorch.org/whl/cu130`, and build args are not
persisted in the image).

Launch pinned to one PRO 5000 GPU UUID (run Docker as root or via `sudo`):

```bash
nvidia-smi --query-gpu=index,uuid,name,memory.total --format=csv
: "${XINFER_GPU_UUID:?Select a GPU UUID}"
XINFER_GPU_UUID="$XINFER_GPU_UUID" bash environments/pro5000/docker/launch.sh
docker exec -it xinfer-pro5000 bash
```

The image sets `NVIDIA_VISIBLE_DEVICES=void` by default; `launch.sh` exposes
only `XINFER_GPU_UUID` as CUDA device 0 with 16 GiB `/dev/shm`, unlimited
`memlock`, 64 MiB stack and no privileged mode. It refuses existing containers
(resume with `docker start xinfer-pro5000 && docker exec -it xinfer-pro5000 bash`).

Environment overrides (defaults shown):

```bash
XINFER_IMAGE=xinfer/pro5000:torch2.13-cu130 XINFER_CONTAINER=xinfer-pro5000 XINFER_HOST_ROOT=/data/xinfer-pro5000 XINFER_GPU_UUID="$XINFER_GPU_UUID"   bash environments/pro5000/docker/launch.sh
```

Host bind mounts under `XINFER_HOST_ROOT`: `workspace` → `/workspace` (place the
repo in `workspace/xinfer`), `models` → `/models`, `cache` → `/cache`,
`reports` → `/reports`. The container runs no SSH daemon; use `docker exec`.

## Check and build inside the container

```bash
nvidia-smi
"$PY" -m environments.check --target pro5000 --dev --output /reports/pro5000-environment.json
bash environments/build.sh --target pro5000 --python "$PY"
"$PY" -m environments.check --target pro5000 --dev --build build
```

For offline runs against checkpoints under `/models`, set
`CARGO_NET_OFFLINE=true` (with a populated `CARGO_HOME`), `HF_HUB_OFFLINE=1` and
`TRANSFORMERS_OFFLINE=1`. Build native targets and export bundles on the target
host, then run [validation](../../../docs/development-tools.md#python-validation-tools)
and record results in [validation status](../../../docs/validation-status.md).
