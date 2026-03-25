#include "common.hpp"
#include <cuda_runtime.h>

vector<int> h_point_cls;
vector<int> h_cls_cnt;
vector<int> h_cls_md_x;
vector<int> h_cls_md_y;
vector<int> h_cls_md_z;

static int* d_x = nullptr;
static int* d_y = nullptr;
static int* d_z = nullptr;
static int* d_point_cls = nullptr;
static int* d_cls_cnt = nullptr;
static long long* d_sum_x = nullptr;
static long long* d_sum_y = nullptr;
static long long* d_sum_z = nullptr;
static int* d_accum_cnt = nullptr;

constexpr int MAX_K = 128;
__constant__ int c_cls_md_x[MAX_K];
__constant__ int c_cls_md_y[MAX_K];
__constant__ int c_cls_md_z[MAX_K];

__device__ inline long long atomicAddLL(long long* address, long long val) {
    unsigned long long* addr = (unsigned long long*)address;
    unsigned long long old = *addr, assumed;
    do {
        assumed = old;
        old = atomicCAS(
            addr,
            assumed,
            (unsigned long long)((long long)assumed + val)
        );
    } while (assumed != old);
    return (long long)old;
}

__global__ void reassign_kernel(int* d_point_cls, int* d_cls_cnt, const int* d_x, const int* d_y, const int* d_z, int n, int k) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        long long min_sq = LLONG_MAX;
        int min_idx = 0;
        for (int j = 0; j < k; j++) {
            long long dx = (long long)c_cls_md_x[j] - d_x[i];
            long long dy = (long long)c_cls_md_y[j] - d_y[i];
            long long dz = (long long)c_cls_md_z[j] - d_z[i];
            long long sq = dx * dx + dy * dy + dz * dz;
            if (sq < min_sq) {
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

static void free_kmeans_device_buffers() {
    CUDA_CHECK(cudaFree(d_x));
    CUDA_CHECK(cudaFree(d_y));
    CUDA_CHECK(cudaFree(d_z));
    CUDA_CHECK(cudaFree(d_point_cls));
    CUDA_CHECK(cudaFree(d_cls_cnt));
    CUDA_CHECK(cudaFree(d_sum_x));
    CUDA_CHECK(cudaFree(d_sum_y));
    CUDA_CHECK(cudaFree(d_sum_z));
    CUDA_CHECK(cudaFree(d_accum_cnt));

    d_x = nullptr;
    d_y = nullptr;
    d_z = nullptr;
    d_point_cls = nullptr;
    d_cls_cnt = nullptr;
    d_sum_x = nullptr;
    d_sum_y = nullptr;
    d_sum_z = nullptr;
    d_accum_cnt = nullptr;
}

static void reassign_cls(bool cpy) {
    int threads_per_block = 256;
    int blocks = (n + threads_per_block - 1) / threads_per_block;

    CUDA_CHECK(cudaMemset(d_cls_cnt, 0, k * sizeof(int)));
    CUDA_CHECK(cudaMemcpyToSymbol(c_cls_md_x, h_cls_md_x.data(), k * sizeof(int)));
    CUDA_CHECK(cudaMemcpyToSymbol(c_cls_md_y, h_cls_md_y.data(), k * sizeof(int)));
    CUDA_CHECK(cudaMemcpyToSymbol(c_cls_md_z, h_cls_md_z.data(), k * sizeof(int)));

    reassign_kernel<<<blocks, threads_per_block>>>(
        d_point_cls, d_cls_cnt, d_x, d_y, d_z, n, k
    );
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    if (cpy) {
        CUDA_CHECK(cudaMemcpy(h_point_cls.data(), d_point_cls, n * sizeof(int), cudaMemcpyDeviceToHost));
    }
    CUDA_CHECK(cudaMemcpy(h_cls_cnt.data(), d_cls_cnt, k * sizeof(int), cudaMemcpyDeviceToHost));
}

__global__ void add_kernel(const int* d_x, const int* d_y, const int* d_z, const int* d_point_cls, long long* d_sum_x, long long* d_sum_y, long long* d_sum_z, int* d_cnt, int n, int k) {
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
            if(h_cls_md_x[i] != md_x || h_cls_md_y[i] != md_y || h_cls_md_z[i] != md_z) changed = true;
            h_cls_md_x[i] = md_x;
            h_cls_md_y[i] = md_y;
            h_cls_md_z[i] = md_z;
        }
    }

    return changed;
}

void kmeans(vector<int>& H) {
    if (k <= 0 || k > MAX_K || n <= 0) return;

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
        reassign_cls(iter==t-1);
        if(!recalculate_md()) {
            CUDA_CHECK(cudaMemcpy(h_point_cls.data(), d_point_cls, n * sizeof(int), cudaMemcpyDeviceToHost));
            break;
        }
    }

    vector<int> cls_hist(1LL * k * 256, 0);
    for (int i = 0; i < n; i++) {
        int cls = h_point_cls[i];
        if (cls >= 0 && cls < k) {
            cls_hist[cls * 256 + intensities[i]] += 1;
        }
    }
    for (int i = 0; i < n; i++) {
        int cls = h_point_cls[i];
        if (cls >= 0 && cls < k) {
            int hbase = i * 256;
            int cbase = cls * 256;
            for (int b = 0; b < 256; b++) {
                H[hbase + b] = cls_hist[cbase + b];
            }
        }
    }

    free_kmeans_device_buffers();
}
