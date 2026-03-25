#include "common.hpp"
#include <cuda_runtime.h>

static constexpr int KMAX = 128;
static constexpr int THREADS_PER_BLOCK = 256;

struct Neighbor {
    int dist;
    int idx;
};

__device__ bool better( const Neighbor& a, const Neighbor& b, const int* x, const int* y, const int* z) {
    if (a.dist != b.dist) return a.dist < b.dist;
    if (x[a.idx] != x[b.idx]) return x[a.idx] < x[b.idx];
    if (y[a.idx] != y[b.idx]) return y[a.idx] < y[b.idx];
    if (z[a.idx] != z[b.idx]) return z[a.idx] < z[b.idx];
    return a.idx < b.idx;
}

__device__ bool worse(const Neighbor& a,const Neighbor& b,const int* x,const int* y,const int* z) {
    return better(b, a, x, y, z);
}

__device__ void bubble_up( Neighbor* heap, int pos, const int* x, const int* y, const int* z) {
    while (pos > 0) {
        int parent = (pos - 1) / 2;
        if (!worse(heap[pos], heap[parent], x, y, z)) break;
        Neighbor tmp = heap[parent];
        heap[parent] = heap[pos];
        heap[pos] = tmp;
        pos = parent;
    }
}

__device__ void bubble_down( Neighbor* heap, int size, int pos, const int* x, const int* y, const int* z) {
    while (true) {
        int left = 2 * pos + 1;
        int right = left + 1;
        int worst = pos;

        if (left < size && worse(heap[left], heap[worst], x, y, z)) worst = left;
        if (right < size && worse(heap[right], heap[worst], x, y, z)) worst = right;

        if (worst == pos) break;
        Neighbor tmp = heap[pos];
        heap[pos] = heap[worst];
        heap[worst] = tmp;
        pos = worst;
    }
}

__global__ void compute_kernel( const int* x, const int* y, const int* z, const int* intens, int n, int k, int* H_flat) {
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

        int dx = (int)xi - x[j];
        int dy = (int)yi - y[j];
        int dz = (int)zi - z[j];

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
    for (int t = 0; t < heap_size; t++) {
        int idx = heap[t].idx;
        int bin = intens[idx];
        if (bin >= 0 && bin < 256) H_flat[hrow + bin] += 1;
    }
}

void knn(vector<vector<int>>& H) {
    int *d_x = nullptr, *d_y = nullptr, *d_z = nullptr;
    int* d_intens = nullptr;
    int* d_H = nullptr;

    CUDA_CHECK(cudaMalloc(&d_x, n * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_y, n * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_z, n * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_intens, n * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_H, 1LL * n * 256 * sizeof(int)));

    CUDA_CHECK(cudaMemcpy(d_x, h_x.data(), n * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_y, h_y.data(), n * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_z, h_z.data(), n * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_intens, intensities.data(), n * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_H, 0, 1LL * n * 256 * sizeof(int)));

    dim3 block_dim(THREADS_PER_BLOCK);
    dim3 grid_dim((n + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK);

    compute_kernel<<<grid_dim, block_dim>>>(d_x,d_y,d_z,d_intens,n,k,d_H);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    vector<int> h_flat_H(1LL * n * 256, 0);
    CUDA_CHECK(cudaMemcpy(
        h_flat_H.data(),
        d_H,
        1LL * n * 256 * sizeof(int),
        cudaMemcpyDeviceToHost
    ));

    for (int i = 0; i < n; i++) {
        int base = i * 256;
        for (int b = 0; b < 256; b++) {
            H[i][b] = h_flat_H[base + b];
        }
    }

    CUDA_CHECK(cudaFree(d_x));
    CUDA_CHECK(cudaFree(d_y));
    CUDA_CHECK(cudaFree(d_z));
    CUDA_CHECK(cudaFree(d_intens));
    CUDA_CHECK(cudaFree(d_H));
}
