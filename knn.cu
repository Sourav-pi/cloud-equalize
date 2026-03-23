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
    int *d_x = nullptr, *d_y = nullptr, *d_z = nullptr;
    long long* d_distsq = nullptr;

    CUDA_CHECK(cudaMalloc(&d_x, n * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_y, n * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_z, n * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_distsq, 1LL * n * n * sizeof(long long)));

    CUDA_CHECK(cudaMemcpy(d_x, h_x.data(), n * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_y, h_y.data(), n * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_z, h_z.data(), n * sizeof(int), cudaMemcpyHostToDevice));

    dim3 block_dim(16, 16);
    dim3 grid_dim((n + block_dim.x - 1) / block_dim.x, (n + block_dim.y - 1) / block_dim.y);

    compute_distances_kernel<<<grid_dim, block_dim>>>(d_x, d_y, d_z, d_distsq, n);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    vector<long long> h_distsq(1LL * n * n);
    CUDA_CHECK(cudaMemcpy(h_distsq.data(), d_distsq, 1LL * n * n * sizeof(long long), cudaMemcpyDeviceToHost));

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            distsq[i][j] = h_distsq[i * n + j];
        }
    }

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

static void print_distances(const vector<vector<long long>>& distsq) {
    for (int i = 0; i < n; i++) {
        cout << "Distances[" << i << "] = { ";
        for (int j = 0; j < n; j++) {
            if (j) cout << ", ";
            cout << distsq[i][j];
        }
        cout << " }\n";
    }
}

void knn(vector<vector<int>>& H) {
    vector<vector<long long>> distsq(n, vector<long long>(n));
    // vector<vector<int>> neighbors(n);
    compute_distances(distsq);
    // print_distances(distsq);

    for (int i = 0; i < n; i++) {
        vector<pair<long long, int>> tmp;
        tmp.reserve(max(0, n - 1));
        for (int j = 0; j < n; j++) {
            if (j != i) tmp.push_back(make_pair(distsq[i][j], j));
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
