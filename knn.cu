#include "common.hpp"
#include <cuda_runtime.h>
static vector<int> h_x;
static vector<int> h_y;
static vector<int> h_z;
static vector<int> intensities;
static int n;
static int k;
static int t;
static string model;
static constexpr int KMAX = 128;
static constexpr int THREADS_PER_BLOCK = 256;
int* d_intensities_knn;
int* d_H_knn;

static void knn_read_input(const string& inpfile) {
    h_x.clear(); h_y.clear(); h_z.clear(); intensities.clear();
    ifstream file(inpfile.c_str());
    file >> n >> k >> t;
    for (int i = 0; i < n; i++) {
        int x, y, z, I;
        file >> x >> y >> z >> I;
        h_x.push_back(x);
        h_y.push_back(y);
        h_z.push_back(z);
        intensities.push_back(I);
    }
    file.close();
}

static void knn_write_output() {
    ofstream file("knn.txt");
    for (int i = 0; i < n; i++) {
        file << h_x[i] << " " << h_y[i] << " " << h_z[i] << " " << intensities[i] << "\n";
    }
    file.close();
}

__device__ __forceinline__ bool better_knn(int cand_dist, int cand_idx, int heap_dist, int heap_idx) {
    if (cand_dist != heap_dist) return cand_dist < heap_dist;
    return cand_idx < heap_idx;
}

__device__ __forceinline__ bool worse_knn(int a_dist, int a_idx, int b_dist, int b_idx) {
    if (a_dist != b_dist) return a_dist > b_dist;
    return a_idx > b_idx;
}

__global__ void knn_build_hist_kernel(const int* x, const int* y, const int* z,
                                      const int* intens, int n, int k, int* H_flat) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    int eff_k = k < (n - 1) ? k : (n - 1);

    int heap_dist[KMAX];
    int heap_idx[KMAX];
    int heap_size = 0;

    int xi = x[i];
    int yi = y[i];
    int zi = z[i];

    for (int j = 0; j < n; j++) {
        if (j == i) continue;

        int dx = xi - x[j];
        int dy = yi - y[j];
        int dz = zi - z[j];
        int cand_dist = dx * dx + dy * dy + dz * dz;

        if (heap_size < eff_k) {
            int p = heap_size;
            heap_dist[p] = cand_dist;
            heap_idx[p] = j;
            heap_size++;

            while (p > 0) {
                int par = (p - 1) >> 1;
                if (!worse_knn(heap_dist[p], heap_idx[p], heap_dist[par], heap_idx[par])) break;
                int td = heap_dist[par]; heap_dist[par] = heap_dist[p]; heap_dist[p] = td;
                int ti = heap_idx[par]; heap_idx[par] = heap_idx[p]; heap_idx[p] = ti;
                p = par;
            }
        } else if (eff_k > 0 && better_knn(cand_dist, j, heap_dist[0], heap_idx[0])) {
            heap_dist[0] = cand_dist;
            heap_idx[0] = j;

            int p = 0;
            while (true) {
                int l = 2 * p + 1;
                int r = l + 1;
                int worst = p;

                if (l < heap_size && worse_knn(heap_dist[l], heap_idx[l], heap_dist[worst], heap_idx[worst])) worst = l;
                if (r < heap_size && worse_knn(heap_dist[r], heap_idx[r], heap_dist[worst], heap_idx[worst])) worst = r;

                if (worst == p) break;
                int td = heap_dist[p]; heap_dist[p] = heap_dist[worst]; heap_dist[worst] = td;
                int ti = heap_idx[p]; heap_idx[p] = heap_idx[worst]; heap_idx[worst] = ti;
                p = worst;
            }
        }
    }

    int hrow = i * 256;
    H_flat[hrow + intens[i]] += 1;
    for (int t = 0; t < heap_size; t++) {
        int bin = intens[heap_idx[t]];
        if (bin >= 0 && bin < 256) H_flat[hrow + bin] += 1;
    }
}

__global__ void knn_compute_cdf_kernel(const int* H, const int* intens,
                                       int* Cmin, int* cdf_at_intensity, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    int base = i * 256;
    int running = 0;
    int cmin = INT_MAX;
    int target = intens[i];
    int target_cdf = 0;

    for (int b = 0; b < 256; b++) {
        running += H[base + b];
        if (running > 0 && cmin == INT_MAX) cmin = running;
        if (b == target) target_cdf = running;
    }

    Cmin[i] = cmin;
    cdf_at_intensity[i] = target_cdf;
}

__global__ void knn_update_intensities_kernel(const int* cdf_at_intensity, const int* Cmin,
                                              int* intens, int n, int k) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    int cmin = Cmin[i];
    int m = k + 1;
    if (m != cmin) {
        int num = cdf_at_intensity[i] - cmin;
        if (num < 0) num = 0;
        intens[i] = (num * 255) / (m - cmin);
    }
}

void knn(const string& inpfile) {
    knn_read_input(inpfile);
    int *d_x = nullptr, *d_y = nullptr, *d_z = nullptr;
    int *d_Cmin = nullptr, *d_cdf_at_intensity = nullptr;
    d_intensities_knn = nullptr;
    d_H_knn = nullptr;

    CUDA_CHECK(cudaMalloc(&d_x, n * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_y, n * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_z, n * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_intensities_knn, n * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_H_knn, 1LL * n * 256 * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_Cmin, n * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_cdf_at_intensity, n * sizeof(int)));

    CUDA_CHECK(cudaMemcpy(d_x, h_x.data(), n * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_y, h_y.data(), n * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_z, h_z.data(), n * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_intensities_knn, intensities.data(), n * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_H_knn, 0, 1LL * n * 256 * sizeof(int)));

    dim3 block_dim(THREADS_PER_BLOCK);
    dim3 grid_dim((n + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK);

    knn_build_hist_kernel<<<grid_dim, block_dim>>>(d_x, d_y, d_z, d_intensities_knn, n, k, d_H_knn);
    CUDA_CHECK(cudaGetLastError());

    knn_compute_cdf_kernel<<<grid_dim, block_dim>>>(d_H_knn, d_intensities_knn, d_Cmin, d_cdf_at_intensity, n);
    CUDA_CHECK(cudaGetLastError());

    knn_update_intensities_kernel<<<grid_dim, block_dim>>>(d_cdf_at_intensity, d_Cmin, d_intensities_knn, n, k);
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaMemcpy(intensities.data(), d_intensities_knn, n * sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaFree(d_x));
    CUDA_CHECK(cudaFree(d_y));
    CUDA_CHECK(cudaFree(d_z));
    CUDA_CHECK(cudaFree(d_H_knn));
    CUDA_CHECK(cudaFree(d_Cmin));
    CUDA_CHECK(cudaFree(d_cdf_at_intensity));
    CUDA_CHECK(cudaFree(d_intensities_knn));

    d_H_knn = nullptr;
    d_intensities_knn = nullptr;

    knn_write_output();
}
