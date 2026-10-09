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
vector<int> h_point_cls;
vector<int> h_cls_cnt;
vector<int> h_cls_md_x;
vector<int> h_cls_md_y;
vector<int> h_cls_md_z;

static int* d_x = nullptr;
static int* d_y = nullptr;
static int* d_z = nullptr;
int* d_point_cls = nullptr;
int* d_cls_cnt = nullptr;
static long long* d_sum_x = nullptr;
static long long* d_sum_y = nullptr;
static long long* d_sum_z = nullptr;
static int* d_accum_cnt = nullptr;
int* d_intensities;
int* d_cls_hist;

static void kmeans_read_input(const string& inpfile) {
    h_x.clear(); h_y.clear(); h_z.clear(); intensities.clear();
    const auto cloud = read_cloud(inpfile, "kmeans");
    n = cloud.n; k = cloud.k; t = cloud.iterations;
    for (int i = 0; i < n; i++) {
        const auto& p = cloud.points[i];
        h_x.push_back(p.x);
        h_y.push_back(p.y);
        h_z.push_back(p.z);
        intensities.push_back(p.intensity);
    }
}

static void kmeans_write_output() {
    ofstream file(output_path);
    if (!file) throw runtime_error("Cannot open output: " + output_path);
    for (int i = 0; i < n; i++) {
        file << h_x[i] << " " << h_y[i] << " " << h_z[i] << " " << intensities[i] << "\n";
    }
    file.close();
    if (!file) throw runtime_error("Failed to write output: " + output_path);
}

constexpr int MAX_K = 128;
__constant__ int c_cls_md_x[MAX_K];
__constant__ int c_cls_md_y[MAX_K];
__constant__ int c_cls_md_z[MAX_K];

__device__ inline long long atomicAddLL(long long* address, long long val) {
    unsigned long long* addr = (unsigned long long*)address;
    unsigned long long old = *addr, assumed;
    do {
        assumed = old;
        old = atomicCAS(addr, assumed, (unsigned long long)((long long)assumed + val));
    } while (assumed != old);
    return (long long)old;
}

__global__ void reassign_kernel(int* d_point_cls, int* d_cls_cnt,
                                const int* d_x, const int* d_y, const int* d_z,
                                int n, int k) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        long long min_sq = LLONG_MAX;
        int min_idx = 0;
        for (int j = 0; j < k; j++) {
            long long dx = (long long)c_cls_md_x[j] - d_x[i];
            long long dy = (long long)c_cls_md_y[j] - d_y[i];
            long long dz = (long long)c_cls_md_z[j] - d_z[i];
            long long sq = dx * dx + dy * dy + dz * dz;
            if (sq < min_sq ||
                (sq == min_sq &&
                 (c_cls_md_x[j] < c_cls_md_x[min_idx] ||
                  (c_cls_md_x[j] == c_cls_md_x[min_idx] &&
                   (c_cls_md_y[j] < c_cls_md_y[min_idx] ||
                    (c_cls_md_y[j] == c_cls_md_y[min_idx] &&
                     c_cls_md_z[j] < c_cls_md_z[min_idx])))))) {
                min_sq = sq;
                min_idx = j;
            }
        }
        d_point_cls[i] = min_idx;
        atomicAdd(&d_cls_cnt[min_idx], 1);
    }
}

static void init_kmeans_device_buffers() {
    CUDA_CHECK(cudaMalloc(&d_x, n * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_y, n * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_z, n * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_point_cls, n * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_cls_cnt, k * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_sum_x, k * sizeof(long long)));
    CUDA_CHECK(cudaMalloc(&d_sum_y, k * sizeof(long long)));
    CUDA_CHECK(cudaMalloc(&d_sum_z, k * sizeof(long long)));
    CUDA_CHECK(cudaMalloc(&d_accum_cnt, k * sizeof(int)));

    CUDA_CHECK(cudaMemcpy(d_x, h_x.data(), n * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_y, h_y.data(), n * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_z, h_z.data(), n * sizeof(int), cudaMemcpyHostToDevice));
}

static void free_kmeans_internal_buffers() {
    CUDA_CHECK(cudaFree(d_x));
    CUDA_CHECK(cudaFree(d_y));
    CUDA_CHECK(cudaFree(d_z));
    CUDA_CHECK(cudaFree(d_sum_x));
    CUDA_CHECK(cudaFree(d_sum_y));
    CUDA_CHECK(cudaFree(d_sum_z));
    CUDA_CHECK(cudaFree(d_accum_cnt));

    d_x = nullptr;
    d_y = nullptr;
    d_z = nullptr;
    d_sum_x = nullptr;
    d_sum_y = nullptr;
    d_sum_z = nullptr;
    d_accum_cnt = nullptr;
}

static void reassign_cls() {
    int threads_per_block = 256;
    int blocks = (n + threads_per_block - 1) / threads_per_block;

    CUDA_CHECK(cudaMemset(d_cls_cnt, 0, k * sizeof(int)));
    CUDA_CHECK(cudaMemcpyToSymbol(c_cls_md_x, h_cls_md_x.data(), k * sizeof(int)));
    CUDA_CHECK(cudaMemcpyToSymbol(c_cls_md_y, h_cls_md_y.data(), k * sizeof(int)));
    CUDA_CHECK(cudaMemcpyToSymbol(c_cls_md_z, h_cls_md_z.data(), k * sizeof(int)));

    reassign_kernel<<<blocks, threads_per_block>>>(d_point_cls, d_cls_cnt, d_x, d_y, d_z, n, k);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

__global__ void add_kernel(const int* d_x, const int* d_y, const int* d_z, const int* d_point_cls,
                           long long* d_sum_x, long long* d_sum_y, long long* d_sum_z,
                           int* d_cnt, int n, int k) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    __shared__ long long s_sum_x[MAX_K];
    __shared__ long long s_sum_y[MAX_K];
    __shared__ long long s_sum_z[MAX_K];
    __shared__ int s_cnt[MAX_K];

    for (int c = threadIdx.x; c < k; c += blockDim.x) {
        s_sum_x[c] = 0;
        s_sum_y[c] = 0;
        s_sum_z[c] = 0;
        s_cnt[c] = 0;
    }
    __syncthreads();

    if (i < n) {
        int cls = d_point_cls[i];
        if (cls >= 0 && cls < k) {
            atomicAddLL(&s_sum_x[cls], (long long)d_x[i]);
            atomicAddLL(&s_sum_y[cls], (long long)d_y[i]);
            atomicAddLL(&s_sum_z[cls], (long long)d_z[i]);
            atomicAdd(&s_cnt[cls], 1);
        }
    }

    __syncthreads();
    for (int c = threadIdx.x; c < k; c += blockDim.x) {
        atomicAddLL(&d_sum_x[c], s_sum_x[c]);
        atomicAddLL(&d_sum_y[c], s_sum_y[c]);
        atomicAddLL(&d_sum_z[c], s_sum_z[c]);
        atomicAdd(&d_cnt[c], s_cnt[c]);
    }
}

static bool recalculate_md() {
    int threads_per_block = 256;
    int blocks = (n + threads_per_block - 1) / threads_per_block;

    CUDA_CHECK(cudaMemset(d_sum_x, 0, k * sizeof(long long)));
    CUDA_CHECK(cudaMemset(d_sum_y, 0, k * sizeof(long long)));
    CUDA_CHECK(cudaMemset(d_sum_z, 0, k * sizeof(long long)));
    CUDA_CHECK(cudaMemset(d_accum_cnt, 0, k * sizeof(int)));

    add_kernel<<<blocks, threads_per_block>>>(d_x, d_y, d_z, d_point_cls, d_sum_x, d_sum_y, d_sum_z, d_accum_cnt, n, k);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    vector<long long> h_sum_x(k, 0), h_sum_y(k, 0), h_sum_z(k, 0);
    vector<int> h_new_cnt(k, 0);
    CUDA_CHECK(cudaMemcpy(h_sum_x.data(), d_sum_x, k * sizeof(long long), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_sum_y.data(), d_sum_y, k * sizeof(long long), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_sum_z.data(), d_sum_z, k * sizeof(long long), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_new_cnt.data(), d_accum_cnt, k * sizeof(int), cudaMemcpyDeviceToHost));
    h_cls_cnt = h_new_cnt;

    bool changed = false;
    for (int i = 0; i < k; i++) {
        if (h_cls_cnt[i] > 0) {
            int md_x = (int)(h_sum_x[i] / h_cls_cnt[i]);
            int md_y = (int)(h_sum_y[i] / h_cls_cnt[i]);
            int md_z = (int)(h_sum_z[i] / h_cls_cnt[i]);
            if (h_cls_md_x[i] != md_x || h_cls_md_y[i] != md_y || h_cls_md_z[i] != md_z) changed = true;
            h_cls_md_x[i] = md_x;
            h_cls_md_y[i] = md_y;
            h_cls_md_z[i] = md_z;
        }
    }

    return changed;
}

__global__ void clust_hist_kernel(int* d_point_cls, int* d_intensities, int* d_cls_hist, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) atomicAdd(&d_cls_hist[d_point_cls[i] * 256 + d_intensities[i]], 1);
}

__global__ void kmeans_compute_terms_kernel(const int* cls_hist, const int* point_cls,
                                            const int* intens, int* Cmin, int* cdf_at_intensity, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    int cls = point_cls[i];
    int base = cls * 256;
    int running = 0;
    int cmin = INT_MAX;
    int target = intens[i];
    int target_cdf = 0;

    for (int b = 0; b < 256; b++) {
        running += cls_hist[base + b];
        if (running > 0 && cmin == INT_MAX) cmin = running;
        if (b == target) target_cdf = running;
    }

    Cmin[i] = cmin;
    cdf_at_intensity[i] = target_cdf;
}

__global__ void kmeans_update_intensity_kernel(const int* cdf_at_intensity, const int* Cmin,
                                                int* intens, const int* cls_cnt,
                                                const int* point_cls, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    int cmin = Cmin[i];
    int m = cls_cnt[point_cls[i]];
    if (m != cmin) {
        int num = cdf_at_intensity[i] - cmin;
        if (num < 0) num = 0;
        intens[i] = (num * 255) / (m - cmin);
    }
}

void kmeans(const string& inpfile) {
    kmeans_read_input(inpfile);
    if (k <= 0 || k > MAX_K || n <= 0) {
        d_point_cls = nullptr;
        d_cls_cnt = nullptr;
        d_intensities = nullptr;
        d_cls_hist = nullptr;
        return;
    }

    if (d_point_cls) { CUDA_CHECK(cudaFree(d_point_cls)); d_point_cls = nullptr; }
    if (d_cls_cnt) { CUDA_CHECK(cudaFree(d_cls_cnt)); d_cls_cnt = nullptr; }
    if (d_intensities) { CUDA_CHECK(cudaFree(d_intensities)); d_intensities = nullptr; }
    if (d_cls_hist) { CUDA_CHECK(cudaFree(d_cls_hist)); d_cls_hist = nullptr; }

    h_point_cls.resize(n);
    h_cls_cnt.resize(k);
    h_cls_md_x.resize(k);
    h_cls_md_y.resize(k);
    h_cls_md_z.resize(k);

    for (int i = 0; i < k; i++) {
        h_cls_md_x[i] = h_x[i];
        h_cls_md_y[i] = h_y[i];
        h_cls_md_z[i] = h_z[i];
    }

    init_kmeans_device_buffers();
    for (int iter = 0; iter < t; iter++) {
        reassign_cls();
        if (!recalculate_md()) break;
    }
    if (t == 0) {
        reassign_cls();
    }

    int threads_per_block = 256;
    int blocks = (n + threads_per_block - 1) / threads_per_block;
    int *d_Cmin = nullptr, *d_cdf_at_intensity = nullptr;

    CUDA_CHECK(cudaMalloc(&d_intensities, n * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_cls_hist, k * 256 * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_Cmin, n * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_cdf_at_intensity, n * sizeof(int)));

    CUDA_CHECK(cudaMemcpy(d_intensities, intensities.data(), n * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_cls_hist, 0, k * 256 * sizeof(int)));

    clust_hist_kernel<<<blocks, threads_per_block>>>(d_point_cls, d_intensities, d_cls_hist, n);
    CUDA_CHECK(cudaGetLastError());

    kmeans_compute_terms_kernel<<<blocks, threads_per_block>>>(d_cls_hist, d_point_cls, d_intensities, d_Cmin, d_cdf_at_intensity, n);
    CUDA_CHECK(cudaGetLastError());

    kmeans_update_intensity_kernel<<<blocks, threads_per_block>>>(d_cdf_at_intensity, d_Cmin, d_intensities, d_cls_cnt, d_point_cls, n);
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaMemcpy(intensities.data(), d_intensities, n * sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaFree(d_Cmin));
    CUDA_CHECK(cudaFree(d_cdf_at_intensity));
    CUDA_CHECK(cudaFree(d_cls_cnt));
    CUDA_CHECK(cudaFree(d_point_cls));
    CUDA_CHECK(cudaFree(d_cls_hist));
    CUDA_CHECK(cudaFree(d_intensities));

    d_cls_cnt = nullptr;
    d_point_cls = nullptr;
    d_cls_hist = nullptr;
    d_intensities = nullptr;

    free_kmeans_internal_buffers();

    kmeans_write_output();
}
