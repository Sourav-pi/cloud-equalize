#include "common.hpp"
#include "knn_heap_utils.cuh"
#include <cuda_runtime.h>

static constexpr int KMAX = 128;
static constexpr int THREADS_PER_BLOCK = 256;
int* d_intensities_knn;
int* d_H_knn;

__global__ void compute_kernel(const int* x, const int* y, const int* z, const int* intens, int n, int k, int* H_flat) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    int eff_k = k < (n - 1) ? k : (n - 1);

    Neighbor heap[KMAX];
    int heap_size = 0;

    int xi = x[i];
    int yi = y[i];
    int zi = z[i];

    for (int j = 0; j < n; j++) {
        if (j == i) continue;

        int dx = xi - x[j];
        int dy = yi - y[j];
        int dz = zi - z[j];

        Neighbor cand;
        cand.dist = dx * dx + dy * dy + dz * dz;
        cand.idx = j;

        if (heap_size < eff_k) {
            heap[heap_size] = cand;
            bubble_up(heap, heap_size, x, y, z);
            heap_size++;
        } else if (eff_k > 0 && better(cand, heap[0], x, y, z)) {
            heap[0] = cand;
            bubble_down(heap, heap_size, 0, x, y, z);
        }
    }

    int hrow = i * 256;
    H_flat[hrow + intens[i]] += 1;
    for (int t = 0; t < heap_size; t++) {
        int idx = heap[t].idx;
        int bin = intens[idx];
        if (bin >= 0 && bin < 256) H_flat[hrow + bin] += 1;
    }
}

void knn() {
    int *d_x = nullptr, *d_y = nullptr, *d_z = nullptr;
    d_intensities_knn = nullptr;
    d_H_knn = nullptr;

    CUDA_CHECK(cudaMalloc(&d_x, n * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_y, n * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_z, n * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_intensities_knn, n * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_H_knn, 1LL * n * 256 * sizeof(int)));

    CUDA_CHECK(cudaMemcpy(d_x, h_x.data(), n * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_y, h_y.data(), n * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_z, h_z.data(), n * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_intensities_knn, intensities.data(), n * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_H_knn, 0, 1LL * n * 256 * sizeof(int)));

    dim3 block_dim(THREADS_PER_BLOCK);
    dim3 grid_dim((n + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK);

    compute_kernel<<<grid_dim, block_dim>>>(d_x, d_y, d_z, d_intensities_knn, n, k, d_H_knn);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // CUDA_CHECK(cudaMemcpy(H.data(),d_H_knn,1LL * n * 256 * sizeof(int),cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_x));
    CUDA_CHECK(cudaFree(d_y));
    CUDA_CHECK(cudaFree(d_z));
    // CUDA_CHECK(cudaFree(d_intensities_knn));
    // CUDA_CHECK(cudaFree(d_H_knn));
}
