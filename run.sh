#!/bin/bash

set -e
cd /home/cse/btech/cs1230751/jobs_sub/A2

module purge

# Try CUDA toolkit module first
module load compiler/cuda/12.3/compilervars || true

# If nvcc still missing, try NVIDIA HPC SDK module
if ! command -v nvcc >/dev/null 2>&1; then
  module load suite/nvidia-hpc-sdk/24.1/cuda12.0 || true
fi

# Final check
if ! command -v nvcc >/dev/null 2>&1; then
  echo "nvcc still not found after module load"
  module list
  exit 127
fi

make clean
make
./sol knn