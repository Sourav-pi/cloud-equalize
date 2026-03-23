#include "common.hpp"
#include <cuda_runtime.h>

static constexpr int TILE_DIM = 16;

__global__ void compute_distances(const int* x, const int* y, const int* z, long long* distsq, int n) {
    if (blockIdx.x < blockIdx.y) return;

    __shared__ int row_x[TILE_DIM], row_y[TILE_DIM], row_z[TILE_DIM];
    __shared__ int col_x[TILE_DIM], col_y[TILE_DIM], col_z[TILE_DIM];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int i = blockIdx.y * TILE_DIM + ty;
    const int j = blockIdx.x * TILE_DIM + tx;

    if (tx == 0 && i < n) {
        row_x[ty] = x[i];
        row_y[ty] = y[i];
        row_z[ty] = z[i];
    }
    if (ty == 0 && j < n) {
        col_x[tx] = x[j];
        col_y[tx] = y[j];
        col_z[tx] = z[j];
    }

    __syncthreads();

    if (i >= n || j >= n || j < i) return;

    long long dx = (long long)row_x[ty] - col_x[tx];
    long long dy = (long long)row_y[ty] - col_y[tx];
    long long dz = (long long)row_z[ty] - col_z[tx];
    long long dist = dx * dx + dy * dy + dz * dz;

    long long idx_ij = 1LL * i * n + j;
    long long idx_ji = 1LL * j * n + i;
    distsq[idx_ij] = dist;
    distsq[idx_ji] = dist;
}

static void compute_distances(vector<long long>& h_distsq) {
    int *d_x = nullptr, *d_y = nullptr, *d_z = nullptr;
    long long* d_distsq = nullptr;

    CUDA_CHECK(cudaMalloc(&d_x, n * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_y, n * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_z, n * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_distsq, 1LL * n * n * sizeof(long long)));

    CUDA_CHECK(cudaMemcpy(d_x, h_x.data(), n * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_y, h_y.data(), n * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_z, h_z.data(), n * sizeof(int), cudaMemcpyHostToDevice));

    dim3 block_dim(TILE_DIM, TILE_DIM);
    dim3 grid_dim((n + TILE_DIM - 1) / TILE_DIM, (n + TILE_DIM - 1) / TILE_DIM);

    compute_distances_kernel<<<grid_dim, block_dim>>>(d_x, d_y, d_z, d_distsq, n);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(h_distsq.data(), d_distsq, 1LL * n * n * sizeof(long long), cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_x));
    CUDA_CHECK(cudaFree(d_y));
    CUDA_CHECK(cudaFree(d_z));
    CUDA_CHECK(cudaFree(d_distsq));
}

static void print_neighbors(const vector<vector<int>>& neighbors) {
    for (int i = 0; i < n; i++) {
        cout << "NN[" << i << "] = { ";
        for (int j = 0; j < (int)neighbors[i].size(); j++) {
            if (j) cout << ", ";
            cout << neighbors[i][j];
        }
        cout << " }\n";
    }
}

static void print_distances(const vector<long long>& distsq) {
    for (int i = 0; i < n; i++) {
        cout << "Distances[" << i << "] = { ";
        for (int j = 0; j < n; j++) {
            if (j) cout << ", ";
            cout << distsq[1LL * i * n + j];
        }
        cout << " }\n";
    }
}

void knn(vector<vector<int>>& H) {
    vector<long long> h_distsq(1LL * n * n, 0);
    compute_distances(h_distsq);
    // print_distances(h_distsq);

    vector<vector<int>> neighbors(n);
    for (int i = 0; i < n; i++) {
        vector<pair<long long, int>> tmp;
        tmp.reserve(max(0, n - 1));
        for (int j = 0; j < n; j++) {
            if (j != i) tmp.push_back(make_pair(h_distsq[1LL * i * n + j], j));
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
        int effective_k = min(k, (int)tmp.size());
        // neighbors[i].reserve(effective_k);
        for (int j = 0; j < effective_k; j++) {
            // neighbors[i].push_back(tmp[j].second);
            H[i][intensities[tmp[j].second]] += 1;
        }
    }

    // print_neighbors(neighbors);
    // print_histograms(H);
}
