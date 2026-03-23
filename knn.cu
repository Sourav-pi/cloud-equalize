#include "common.hpp"
#include <cuda_runtime.h>

static constexpr int TILE_DIM = 16;
static constexpr int BATCH_SIZE = 128;

__global__ void compute_distances_kernel(
    const int* x,
    const int* y,
    const int* z,
    long long* batch_distsq,
    int n,
    int row_start,
    int batch_rows
) {
    __shared__ int row_x[TILE_DIM], row_y[TILE_DIM], row_z[TILE_DIM];
    __shared__ int col_x[TILE_DIM], col_y[TILE_DIM], col_z[TILE_DIM];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;

    const int local_i = blockIdx.y * TILE_DIM + ty;
    const int j = blockIdx.x * TILE_DIM + tx;

    if (tx == 0 && local_i < batch_rows) {
        int i_global = row_start + local_i;
        row_x[ty] = x[i_global];
        row_y[ty] = y[i_global];
        row_z[ty] = z[i_global];
    }
    if (ty == 0 && j < n) {
        col_x[tx] = x[j];
        col_y[tx] = y[j];
        col_z[tx] = z[j];
    }

    __syncthreads();

    if (local_i >= batch_rows || j >= n) return;

    long long dx = (long long)row_x[ty] - col_x[tx];
    long long dy = (long long)row_y[ty] - col_y[tx];
    long long dz = (long long)row_z[ty] - col_z[tx];
    long long dist = dx * dx + dy * dy + dz * dz;

    batch_distsq[1LL * local_i * n + j] = dist;
}

static void compute_distances_batch(
    int row_start,
    int batch_rows,
    const int* d_x,
    const int* d_y,
    const int* d_z,
    long long* d_batch_distsq,
    vector<long long>& h_batch_distsq
) {
    dim3 block_dim(TILE_DIM, TILE_DIM);
    dim3 grid_dim((n + TILE_DIM - 1) / TILE_DIM, (batch_rows + TILE_DIM - 1) / TILE_DIM);

    compute_distances_kernel<<<grid_dim, block_dim>>>(
        d_x, d_y, d_z, d_batch_distsq, n, row_start, batch_rows
    );
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(
        h_batch_distsq.data(),
        d_batch_distsq,
        1LL * batch_rows * n * sizeof(long long),
        cudaMemcpyDeviceToHost
    ));
}

void knn(vector<vector<int>>& H) {
    int *d_x = nullptr, *d_y = nullptr, *d_z = nullptr;
    long long* d_batch_distsq = nullptr;

    CUDA_CHECK(cudaMalloc(&d_x, n * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_y, n * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_z, n * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_batch_distsq, 1LL * BATCH_SIZE * n * sizeof(long long)));

    CUDA_CHECK(cudaMemcpy(d_x, h_x.data(), n * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_y, h_y.data(), n * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_z, h_z.data(), n * sizeof(int), cudaMemcpyHostToDevice));

    for (int row_start = 0; row_start < n; row_start += BATCH_SIZE) {
        int batch_rows = min(BATCH_SIZE, n - row_start);
        vector<long long> h_batch_distsq(1LL * batch_rows * n);

        compute_distances_batch(
            row_start,
            batch_rows,
            d_x,
            d_y,
            d_z,
            d_batch_distsq,
            h_batch_distsq
        );

        for (int bi = 0; bi < batch_rows; bi++) {
            int i = row_start + bi;

            vector<pair<long long, int>> tmp;
            tmp.reserve(max(0, n - 1));
            for (int j = 0; j < n; j++) {
                if (j != i) tmp.push_back(make_pair(h_batch_distsq[1LL * bi * n + j], j));
            }

            sort(tmp.begin(), tmp.end(), [](const pair<long long, int>& a, const pair<long long, int>& b) {
                if (a.first != b.first) return a.first < b.first;

                int ia = a.second;
                int ib = b.second;
                if (h_x[ia] != h_x[ib]) return h_x[ia] < h_x[ib];
                if (h_y[ia] != h_y[ib]) return h_y[ia] < h_y[ib];
                if (h_z[ia] != h_z[ib]) return h_z[ia] < h_z[ib];
                return ia < ib;
            });

            int eff_k = min(k, (int)tmp.size());
            for (int j = 0; j < eff_k; j++) {
                H[i][intensities[tmp[j].second]] += 1;
            }
        }
    }

    CUDA_CHECK(cudaFree(d_x));
    CUDA_CHECK(cudaFree(d_y));
    CUDA_CHECK(cudaFree(d_z));
    CUDA_CHECK(cudaFree(d_batch_distsq));
}
