#!/bin/bash -l

set -euo pipefail
cd /home/cse/btech/cs1230751/jobs_sub/COL380-A2

# Optional overrides when submitting/rerunning:
#   INPUT_FILE=input.txt RUNS=5 WARMUP=1 MODELS="knn approx_knn kmeans" ./run.sh
INPUT_FILE="${INPUT_FILE:-input.txt}"
RUNS="${RUNS:-1}"
WARMUP="${WARMUP:-0}"
MODELS="${MODELS:-knn}"
# MODELS="${MODELS:-knn approx_knn kmeans}"

module purge
module load compiler/gcc/9.1.0
module load suite/nvidia-hpc-sdk/20.7/cuda11.0

nvidia-smi
nvcc --version
g++ --version

make clean
make NVCCFLAGS='-O2 -std=c++11 -x cu -arch=sm_35 -Xcompiler -fopenmp'
make cpu

# python benchmark_speedup.py \
# 	--input "${INPUT_FILE}" \
# 	--models ${MODELS} \
# 	--runs "${RUNS}" \
# 	--warmup "${WARMUP}" \
# 	--no-build
./sol input.txt knn
