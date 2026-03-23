#!/bin/bash -l

set -euo pipefail
cd /home/cse/btech/cs1230751/jobs_sub/COL380-A2

module purge
module load compiler/gcc/9.1.0
module load suite/nvidia-hpc-sdk/20.7/cuda11.0

nvidia-smi
nvcc --version

make clean
make NVCCFLAGS='-O2 -std=c++11 -x cu -arch=sm_35 -Xcompiler -fopenmp'
./sol input.txt knn
