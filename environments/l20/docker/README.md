# L20 Docker testing environment

This image builds directly from `nvidia/cuda:12.6.3-devel-ubuntu24.04` (CUDA
12.6.3 toolkit, `nvcc` 12.6.85, GCC 13.2, `ncu`) with Python 3.12,
a minimal Rust toolchain (`/opt/rust`), and a single virtual environment at
`/opt/xinfer/venv` (`$PY`, first on `PATH`). Python and build dependencies come
straight from [environments/l20/requirements.txt](../requirements.txt) (which
includes [environments/requirements.txt](../../requirements.txt)) and
[requirements-dev.txt](../../../requirements-dev.txt)—no separate Docker lock
files are maintained.

Project source, model checkpoints, caches and reports live in host bind mounts.

## Build and launch

Run `docker build` from the repository root so Docker can read the shared
requirements files (root `.dockerignore` excludes `.git`, `work`, `build` and
`artifacts`):

```bash
docker build -f environments/l20/docker/Dockerfile \
  -t xinfer/l20:torch2.13-cu126 .
```

Pass `--build-arg PIP_INDEX_URL=<index-url>` at build time if you want pip to
use a regional PyPI mirror; the Torch/CUDA index remains
`https://download.pytorch.org/whl/cu126` from `environments/l20/requirements.txt`,
and build arguments are not persisted into the final image.

Launch a container pinned to a single L20 GPU UUID:

```bash
nvidia-smi --query-gpu=index,uuid,name,memory.total --format=csv
: "${XINFER_GPU_UUID:?Select a GPU UUID}"
XINFER_GPU_UUID="$XINFER_GPU_UUID" bash environments/l20/docker/launch.sh
docker exec -it xinfer-l20 bash
```

Run host Docker commands as root or through `sudo`. The launcher requires an
explicit GPU UUID (exposed as CUDA device 0 inside the container), leaves CPU
and RAM limits unset, configures 16 GiB shared memory, unlimited locked memory
and a 64 MiB stack, and needs no privileged container. The image sets
`NVIDIA_VISIBLE_DEVICES=void` by default so ad-hoc `docker run` calls do not
accidentally expose all host GPUs.

`launch.sh` accepts environment overrides:

```bash
XINFER_IMAGE=xinfer/l20:torch2.13-cu126 \
XINFER_CONTAINER=xinfer-l20 \
XINFER_HOST_ROOT=/data/xinfer-l20 \
XINFER_GPU_UUID="$XINFER_GPU_UUID" \
  bash environments/l20/docker/launch.sh
```

It refuses to replace an existing container; use `docker start xinfer-l20` and
`docker exec` to resume it. Persistent host bind mounts are:

| Host path below `XINFER_HOST_ROOT` | Container path |
| --- | --- |
| `workspace` | `/workspace` |
| `models` | `/models` |
| `cache` | `/cache` |
| `reports` | `/reports` |

Place the repository in `workspace/xinfer`. The image does not run an SSH
server; execute commands through `docker exec`.

## Check and build the mounted project

Inside the running container, `python` and `$PY` point to
`/opt/xinfer/venv/bin/python` (serving Qwen3, Qwen3.5, Qwen3.8, native CMake/Ninja
builds, `ruff` and `clang-format`):

```bash
nvidia-smi
"$PY" -c 'import torch; print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0), torch.cuda.mem_get_info())'
"$PY" -m environments.check --target l20 --dev \
  --output /reports/l20-environment.json

bash environments/build.sh --target l20 --python "$PY"
"$PY" -m environments.check --target l20 --dev --build build
```

Set `CARGO_NET_OFFLINE=true` (with a pre-populated `CARGO_HOME` cache) and
`HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1` when running against
pre-downloaded checkpoints under `/models`. Freshly build native targets and
model packages on the target host rather than copying generated `.so` files
across hosts with different CPU instruction sets.

After launch, validate actual Triton GPU compilation/execution, AOTI packaging,
Python/native execution and shared-weight loading using the
[shared validation tools](../../../docs/development-tools.md#python-validation-tools)
and record tested profiles in [validation status](../../../docs/validation-status.md).
