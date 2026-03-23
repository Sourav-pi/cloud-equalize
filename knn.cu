#include "common.hpp"
#include <cuda_runtime.h>

__global__ void compute_distances_kernel(int* x, int* y, int* z, long long* distsq, int n) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n && j < n) {
        long long dx = (long long)x[i] - x[j];
        long long dy = (long long)y[i] - y[j];
        long long dz = (long long)z[i] - z[j];
        distsq[i * n + j] = dx * dx + dy * dy + dz * dz;
    }
}

static void compute_distances(vector<vector<long long>>& distsq) {
    int *d_x, *d_y, *d_z;
    long long* d_distsq;

    cudaMalloc(&d_x, n * sizeof(int));
    cudaMalloc(&d_y, n * sizeof(int));
    cudaMalloc(&d_z, n * sizeof(int));
    cudaMalloc(&d_distsq, n * n * sizeof(long long));

    cudaMemcpy(d_x, h_x.data(), n * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, h_y.data(), n * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_z, h_z.data(), n * sizeof(int), cudaMemcpyHostToDevice);

    dim3 block_dim(16, 16);
    dim3 grid_dim((n + block_dim.x - 1) / block_dim.x, (n + block_dim.y - 1) / block_dim.y);

    compute_distances_kernel<<<grid_dim, block_dim>>>(d_x, d_y, d_z, d_distsq, n);

    long long* h_distsq = (long long*)malloc(n * n * sizeof(long long));
    cudaMemcpy(h_distsq, d_distsq, n * n * sizeof(long long), cudaMemcpyDeviceToHost);

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            distsq[i][j] = h_distsq[i * n + j];
        }
    }

    cudaFree(d_x);
    cudaFree(d_y);
    cudaFree(d_z);
    cudaFree(d_distsq);
    free(h_distsq);
}

void knn(vector<vector<int>>& H) {
    vector<vector<long long>> distsq(n, vector<long long>(n));
    compute_distances(distsq);

    for (int i = 0; i < n; i++) {
        vector<pair<long long, int>> tmp;
        tmp.reserve(max(0, n - 1));
        for (int j = 0; j < n; j++) {
            if (j != i) tmp.push_back(make_pair(distsq[i][j], j));
        }
        sort(tmp.begin(), tmp.end());
        for (int j = 0; j < k; j++) {
            H[i][intensities[tmp[j].second]] += 1;
        }
    }

    print_histograms(H);

    
}
