#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

# Set site-specific module names through the environment before running.
if [[ -n "${COMPILER_MODULE:-}" ]]; then module load "$COMPILER_MODULE"; fi
if [[ -n "${CUDA_MODULE:-}" ]]; then module load "$CUDA_MODULE"; fi
nvidia-smi
make all CUDA_ARCH="${CUDA_ARCH:-native}"
python3 scripts/validate.py --input "${INPUT_FILE:-examples/sample.txt}" --backend both
python3 scripts/benchmark.py --input "${INPUT_FILE:-examples/sample.txt}" \
  --runs "${RUNS:-5}" --warmup "${WARMUP:-1}" \
  --output "${RESULT_FILE:-results/cluster.json}"
