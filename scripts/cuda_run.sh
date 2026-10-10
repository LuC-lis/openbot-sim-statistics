#!/usr/bin/env bash
# Run a Python script with the CUDA runtime libraries from the pip
# `nvidia-*-cu12` wheels made visible to TensorFlow. This is only needed
# if you installed TensorFlow 2.15 with pip + nvidia-*-cu12 packages.
#
# Usage:
#   ./scripts/cuda_run.sh scripts/train.py --epochs 15
#
set -euo pipefail
PY="${PYTHON:-python3}"
SITE=$("$PY" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')
PREFIX=$(dirname "$SITE")

NV=""
for d in "$SITE"/nvidia/*/lib; do [ -d "$d" ] && NV="$NV$d:"; done
export LD_LIBRARY_PATH="${NV}${PREFIX}/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
NVCC=$(ls -d "$SITE"/nvidia/cuda_nvcc 2>/dev/null | head -1 || true)
[ -n "${NVCC:-}" ] && export XLA_FLAGS="--xla_gpu_cuda_data_dir=${NVCC} ${XLA_FLAGS:-}"
exec "$PY" "$@"
