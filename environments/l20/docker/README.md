# L20 Docker testing environment

This image provides Python 3.12.13, Torch `2.13.0+cu126`, Triton `3.7.1`, the
native build tools and separate Qwen3 / Qwen3.5 model dependencies. The pinned
TencentOS 3.2 base supplies GCC 11.2 and CUDA 12.4.131. Native builds explicitly
link the CUDA 12.6 runtime supplied with Torch.

`XINFER_CUDA_LIBS` selects the CUDA libraries installed with the pinned Torch
stack before the older SDK libraries. `LD_LIBRARY_PATH` applies that selection
when programs run, and `LDFLAGS` supplies the same directories through
`-rpath-link` when native programs link. This also selects CUDA 12.6 CUPTI, whose
newer entry points are required by LibTorch. NVCC and headers remain from the
12.4 SDK; the container runtime supplies the host GPU driver.

Project source, model checkpoints, generated packages and reports live in host
bind mounts. The final image contains the installed environment and the small
build recipe; wheel archives remain in the offline build context and build cache.

## Prepare and transfer the offline context

Prepare the following files on a machine that can obtain the required packages:

```text
Dockerfile
.dockerignore
README.md
launch.sh
native.lock
qwen3.lock
qwen35.lock
SHA256SUMS
payload/
  python-runtime.tar.gz
  nlohmann-json.tar.xz
  wheels/*.whl
```

`python-runtime.tar.gz` contains the standalone CPython 3.12.13 installation
under one top-level directory. The build strips that directory and creates a
clean native package installation under `/opt/xinfer/native`.

`native.lock` records the tested Torch, Triton and CUDA package closure plus
CMake `4.3.4`, Ninja `1.13.2`, Ruff `0.17.3` and clang-format `23.1.3`.
`qwen3.lock` and `qwen35.lock` record each model environment's complete package
freeze. Their native packages must match `native.lock`; model installation uses
that file as constraints. Download Linux x86-64 CPython 3.12 wheels for every
required dependency. The model environments share the controlled native prefix
through `--system-site-packages` and keep their HF packages isolated.

Use nlohmann/json `3.11.3`'s `json.tar.xz`, with SHA-256
`d6c65aca6b1ed68e7a182f4757257b107ae403032760ed6ef121c9d55e81757d`.
Its source tree is installed in `/opt/xinfer/deps/nlohmann_json` for offline
CMake configuration.

Generate `SHA256SUMS` over the three lock files and all payload files, using
paths relative to this context directory. Keep recipe files outside this
manifest so documentation and final image settings can change without
invalidating the installed dependency layers.
The Docker build verifies the complete manifest before unpacking or installing.

Transfer the context archive directly to the target host. When access goes
through a jump host, stream the transfer to the target without staging files
on the jump host. Keep host addresses and access instructions outside the
repository.

## Build and launch

The pinned base must already be present in Docker's local image store. Transfer
and load it separately if needed; `docker build --network=none` controls build
steps, so checking the base first also avoids an implicit registry lookup.
Set `XINFER_BASE_IMAGE` to a locally available image reference pinned by digest.
The tested TencentOS 3.2 / CUDA 12.4.131 base has digest
`sha256:e399ac1af0550a0330783c933f1ebbf46fffc4ef92b302671a76f1d882ddebaa`.
Keep registry addresses and host-specific settings outside the repository.

```bash
cd /data/xinfer-l20/image
: "${XINFER_BASE_IMAGE:?Set a local base image reference pinned by digest}"
docker image inspect "$XINFER_BASE_IMAGE"
sha256sum -c SHA256SUMS
docker build --network=none --pull=false \
  --build-arg XINFER_BASE_IMAGE="$XINFER_BASE_IMAGE" \
  -t xinfer/l20:torch2.13-cu126 .
nvidia-smi --query-gpu=index,uuid,name,memory.total --format=csv
# Set XINFER_GPU_UUID to the selected device before launch.
: "${XINFER_GPU_UUID:?Select a GPU UUID}"
XINFER_GPU_UUID="$XINFER_GPU_UUID" bash launch.sh
docker exec -it xinfer-l20 bash
```

Run the host commands as root or through sudo. The launcher requires an explicit
GPU UUID; that single device appears as CUDA device 0 inside the container.
It leaves CPU and RAM limits unset and provides 16 GiB shared memory, unlimited locked memory and
a 64 MiB stack. It requires no privileged container. Networking uses Docker's
bridge; pip and Hugging Face default to offline operation.

The launcher accepts environment overrides:

```bash
XINFER_IMAGE=xinfer/l20:torch2.13-cu126 \
XINFER_CONTAINER=xinfer-l20 \
XINFER_HOST_ROOT=/data/xinfer-l20 \
XINFER_GPU_UUID="$XINFER_GPU_UUID" \
  bash launch.sh
```

It refuses to replace an existing container. Use `docker start xinfer-l20` and
`docker exec` to resume it. Persistent paths are:

| Host path below `XINFER_HOST_ROOT` | Container path |
| --- | --- |
| `workspace` | `/workspace` |
| `models` | `/models` |
| `cache` | `/cache` |
| `reports` | `/reports` |

Place the repository in `workspace/xinfer`. The image does not start an SSH
server; execute commands through Docker on the host.

## Check and build the mounted project

Inside the running container, `python` defaults to Qwen3.5. Use `$Q3_PY` or
`$Q35_PY` explicitly for each model and `$NATIVE_PY` for native-stack checks.
The base image's Python 3.9 and user site packages are outside these environments.

```bash
nvidia-smi
"$NATIVE_PY" -c 'import torch; print(torch.__version__, torch.version.cuda); print(torch.cuda.get_device_name()); print(torch.cuda.mem_get_info())'
bash environments/check.sh --python "$Q3_PY" --target l20 --model qwen3 \
  --output /reports/qwen3-environment.json
bash environments/check.sh --python "$Q35_PY" --target l20 --model qwen35 \
  --output /reports/qwen35-environment.json

TORCH_PREFIX=$("$NATIVE_PY" -c 'import torch; print(torch.utils.cmake_prefix_path)')
CUDA_RUNTIME=$("$NATIVE_PY" -c 'from pathlib import Path; import torch; print(Path(torch.__file__).resolve().parent.parent / "nvidia/cuda_runtime/lib/libcudart.so.12")')
for MODEL in qwen3 qwen35; do
  "$CMAKE" -S . -B "build/$MODEL" -G Ninja \
    -DCMAKE_PREFIX_PATH="$TORCH_PREFIX" \
    -DCMAKE_CUDA_COMPILER="$CUDA_HOME/bin/nvcc" \
    -DCUDA_cudart_LIBRARY="$CUDA_RUNTIME" \
    -DCUDA_CUDART_LIBRARY="$CUDA_RUNTIME" \
    -DFETCHCONTENT_SOURCE_DIR_NLOHMANN_JSON="$FETCHCONTENT_SOURCE_DIR_NLOHMANN_JSON" \
    -DFETCHCONTENT_FULLY_DISCONNECTED=ON \
    -DTORCH_CUDA_ARCH_LIST=8.9 \
    -DCMAKE_BUILD_TYPE=Release \
    -DXINFER_BUILD_VALIDATION=ON -DXINFER_BUILD_BENCHMARK=ON
  "$CMAKE" --build "build/$MODEL" -j2
done
```

CMake requires the explicit `-DFETCHCONTENT_SOURCE_DIR_NLOHMANN_JSON` argument;
the matching environment variable is provided for that command. Freshly build
the native targets and model packages on the target host rather than copying generated
SO files from a host with different CPU instructions.

After launch, validate actual Triton GPU compilation/execution, AOTI packaging,
Python/native execution and shared-weight loading. Build-time CPU imports and
dependency checks do not establish GPU compatibility. Record CUDA's reported
memory capacity and the container's memory configuration with those checks.
