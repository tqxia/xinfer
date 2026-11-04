#!/usr/bin/env bash
set -euo pipefail

if [[ $# != 0 ]]; then
  printf 'Configure this launcher through XINFER_IMAGE, XINFER_CONTAINER, XINFER_HOST_ROOT and XINFER_GPU_UUID.\n' >&2
  exit 2
fi

image="${XINFER_IMAGE:-xinfer/l20:torch2.13-cu126}"
container="${XINFER_CONTAINER:-xinfer-l20}"
host_root="${XINFER_HOST_ROOT:-/data/xinfer-l20}"
gpu_uuid="${XINFER_GPU_UUID:?Select a GPU UUID with XINFER_GPU_UUID}"

case "$host_root" in
  /*) ;;
  *) printf 'XINFER_HOST_ROOT must be an absolute path.\n' >&2; exit 2 ;;
esac
if [[ "$host_root" == *','* || "$host_root" == *$'\n'* ]]; then
  printf 'XINFER_HOST_ROOT cannot contain commas or newlines.\n' >&2
  exit 2
fi

docker image inspect "$image" >/dev/null
if docker container inspect "$container" >/dev/null 2>&1; then
  printf 'Container %s already exists; use docker start/exec, or select XINFER_CONTAINER.\n' "$container" >&2
  exit 1
fi
nvidia-smi -i "$gpu_uuid" --query-gpu=uuid,name,memory.total --format=csv,noheader

mkdir -p "$host_root/workspace/xinfer" "$host_root/models" "$host_root/cache/tmp" "$host_root/reports"

# No CPU or device-memory quota: GPU selection isolates this testing environment.
exec docker run --detach \
  --name "$container" \
  --user 0:0 \
  --gpus "device=$gpu_uuid" \
  --env "NVIDIA_VISIBLE_DEVICES=$gpu_uuid" \
  --network bridge \
  --shm-size 16g \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  --mount "type=bind,src=$host_root/workspace,dst=/workspace" \
  --mount "type=bind,src=$host_root/models,dst=/models" \
  --mount "type=bind,src=$host_root/cache,dst=/cache" \
  --mount "type=bind,src=$host_root/reports,dst=/reports" \
  --workdir /workspace/xinfer \
  --label xinfer.environment=l20-testing \
  "$image"
