# L20 Docker testing environment

Builds from `nvidia/cuda:12.6.3-devel-ubuntu24.04` (CUDA 12.6.3, GCC 13.2,
`ncu`) with Python 3.12, a minimal Rust toolchain (`/opt/rust`), and one venv at
`/opt/xinfer/venv` (`$PY`, first on `PATH`) installed directly from
[environments/l20/requirements.txt](../requirements.txt) and
[requirements-dev.txt](../../../requirements-dev.txt). Source, checkpoints, caches
and reports live in host bind mounts.

## Build and launch

Build from the repository root (`.dockerignore` excludes `.git`, `work`, `build`
and `artifacts`):

```bash
docker build -f environments/l20/docker/Dockerfile   -t xinfer/l20:torch2.13-cu126 .
```

Pass `--build-arg PIP_INDEX_URL=<index-url>` to use a regional PyPI mirror
(Torch wheels still pull from `https://download.pytorch.org/whl/cu126`, and
build args are not persisted in the image).

Launch pinned to one L20 GPU UUID (run Docker as root or via `sudo`):

```bash
nvidia-smi --query-gpu=index,uuid,name,memory.total --format=csv
: "${XINFER_GPU_UUID:?Select a GPU UUID}"
XINFER_GPU_UUID="$XINFER_GPU_UUID" bash environments/l20/docker/launch.sh
docker exec -it xinfer-l20 bash
```

The image sets `NVIDIA_VISIBLE_DEVICES=void` by default; `launch.sh` exposes
only `XINFER_GPU_UUID` as CUDA device 0 with 16 GiB `/dev/shm`, unlimited
`memlock`, 64 MiB stack and no privileged mode. It refuses existing containers
(resume with `docker start xinfer-l20 && docker exec -it xinfer-l20 bash`).

Environment overrides (defaults shown):

```bash
XINFER_IMAGE=xinfer/l20:torch2.13-cu126 XINFER_CONTAINER=xinfer-l20 XINFER_HOST_ROOT=/data/xinfer-l20 XINFER_GPU_UUID="$XINFER_GPU_UUID"   bash environments/l20/docker/launch.sh
```

Host bind mounts under `XINFER_HOST_ROOT`: `workspace` → `/workspace` (place the
repo in `workspace/xinfer`), `models` → `/models`, `cache` → `/cache`,
`reports` → `/reports`. The container runs no SSH daemon; use `docker exec`.

## Check and build inside the container

```bash
nvidia-smi
"$PY" -m environments.check --target l20 --dev --output /reports/l20-environment.json
bash environments/build.sh --target l20 --python "$PY"
"$PY" -m environments.check --target l20 --dev --build build
```

For offline runs against checkpoints under `/models`, set
`CARGO_NET_OFFLINE=true` (with a populated `CARGO_HOME`), `HF_HUB_OFFLINE=1` and
`TRANSFORMERS_OFFLINE=1`. Build native targets and export bundles on the target
host, then run [validation](../../../docs/development-tools.md#python-validation-tools)
and record results in [validation status](../../../docs/validation-status.md).
