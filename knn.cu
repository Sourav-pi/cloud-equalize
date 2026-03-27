#include "common.hpp"
#include <cuda_runtime.h>
#include <fstream>
#include <string>

using namespace std;

struct Point {
    short x;
    short y;
    short z;
    uint8_t intensity;
    uint8_t padding; // Keep struct size aligned to 8 bytes for clean memory loads
};

// Replaced vector with a raw pointer for pinned memory
static Point* h_points = nullptr;
static int n;
static int k;
static int t;
static string model;
static constexpr int KMAX = 128;
static constexpr int THREADS_PER_BLOCK = 256;

Point* d_points;
int* d_H_knn;

static void knn_read_input(const string& inpfile) {
    ifstream file(inpfile.c_str());
    file >> n >> k >> t;
    
    // Allocate pinned host memory now that we know 'n'
    CUDA_CHECK(cudaMallocHost(&h_points, n * sizeof(Point)));
    
    // Read directly into the pre-allocated pinned memory array
    for (int i = 0; i < n; i++) {
        file >> h_points[i].x >> h_points[i].y >> h_points[i].z >> h_points[i].intensity;
    }
    file.close();
}

static void knn_write_output() {
    ofstream file("knn.txt");
    for (int i = 0; i < n; i++) {
        file << h_points[i].x << " " << h_points[i].y << " " << h_points[i].z << " " << h_points[i].intensity << "\n";
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

// TILED KERNEL
__global__ void knn_build_hist_kernel(const Point* points, int n, int k, int* H_flat) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Allocate shared memory for the tile
    __shared__ Point shared_points[THREADS_PER_BLOCK];

    int eff_k = k < (n - 1) ? k : (n - 1);

    int heap_dist[KMAX];
    int heap_idx[KMAX];
    int heap_size = 0;

    // Load the target point for this specific thread into registers
    int xi = 0, yi = 0, zi = 0;
    if (i < n) {
        xi = points[i].x;
        yi = points[i].y;
        zi = points[i].z;
    }

    // Calculate how many tiles we need to process all 'n' points
    int num_tiles = (n + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    
    #pragma unroll 4
    for (int tile = 0; tile < num_tiles; ++tile) {
        // 1. Load data into shared memory
        int global_idx = tile * THREADS_PER_BLOCK + threadIdx.x;
        if (global_idx < n) {
            shared_points[threadIdx.x] = points[global_idx];
        }
        
        // Wait for all threads in the block to finish loading the tile
        __syncthreads();

        // Only compute if this thread represents a valid target point
        if (i < n) {
            // 2. Compute distances against the tile
            int num_elements_in_tile = (tile == num_tiles - 1) ? (n - tile * THREADS_PER_BLOCK) : THREADS_PER_BLOCK;
            
            #pragma unroll
            for (int j = 0; j < num_elements_in_tile; ++j) {
                int cand_idx = tile * THREADS_PER_BLOCK + j;
                if (cand_idx == i) continue;

                // Read from shared memory instead of global memory!
                int dx = xi - shared_points[j].x;
                int dy = yi - shared_points[j].y;
                int dz = zi - shared_points[j].z;
                int cand_dist = dx * dx + dy * dy + dz * dz;

                // Heap update logic remains identical
                if (heap_size < eff_k) {
                    int p = heap_size;
                    heap_dist[p] = cand_dist;
                    heap_idx[p] = cand_idx;
                    heap_size++;

                    while (p > 0) {
                        int par = (p - 1) >> 1;
                        if (!worse_knn(heap_dist[p], heap_idx[p], heap_dist[par], heap_idx[par])) break;
                        int td = heap_dist[par]; heap_dist[par] = heap_dist[p]; heap_dist[p] = td;
                        int ti = heap_idx[par]; heap_idx[par] = heap_idx[p]; heap_idx[p] = ti;
                        p = par;
                    }
                } else if (eff_k > 0 && better_knn(cand_dist, cand_idx, heap_dist[0], heap_idx[0])) {
                    heap_dist[0] = cand_dist;
                    heap_idx[0] = cand_idx;

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
        }
        
        // Wait for all threads to finish computing before loading the next tile
        __syncthreads();
    }

    // Write results to histogram (only for valid threads)
    if (i < n) {
        int hrow = i * 256;
        H_flat[hrow + points[i].intensity] += 1;
        for (int t = 0; t < heap_size; t++) {
            int bin = points[heap_idx[t]].intensity;
            if (bin >= 0 && bin < 256) H_flat[hrow + bin] += 1;
        }
    }
}

__global__ void knn_compute_cdf_kernel(const int* H, const Point* points,
                                       int* Cmin, int* cdf_at_intensity, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    int base = i * 256;
    int running = 0;
    int cmin = INT_MAX;
    int target = points[i].intensity;
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
                                              Point* points, int n, int k) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    int cmin = Cmin[i];
    int m = k + 1;
    if (m != cmin) {
        int num = cdf_at_intensity[i] - cmin;
        if (num < 0) num = 0;
        points[i].intensity = (num * 255) / (m - cmin);
    }
}

void knn(const string& inpfile) {
    knn_read_input(inpfile);
    
    int *d_Cmin = nullptr, *d_cdf_at_intensity = nullptr;
    d_points = nullptr;
    d_H_knn = nullptr;

    CUDA_CHECK(cudaMalloc(&d_points, n * sizeof(Point)));
    CUDA_CHECK(cudaMalloc(&d_H_knn, 1LL * n * 256 * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_Cmin, n * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_cdf_at_intensity, n * sizeof(int)));

    // Notice we pass h_points directly now, without .data()
    CUDA_CHECK(cudaMemcpy(d_points, h_points, n * sizeof(Point), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_H_knn, 0, 1LL * n * 256 * sizeof(int)));

    dim3 block_dim(THREADS_PER_BLOCK);
    dim3 grid_dim((n + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK);

    knn_build_hist_kernel<<<grid_dim, block_dim>>>(d_points, n, k, d_H_knn);
    CUDA_CHECK(cudaGetLastError());

    knn_compute_cdf_kernel<<<grid_dim, block_dim>>>(d_H_knn, d_points, d_Cmin, d_cdf_at_intensity, n);
    CUDA_CHECK(cudaGetLastError());

    knn_update_intensities_kernel<<<grid_dim, block_dim>>>(d_cdf_at_intensity, d_Cmin, d_points, n, k);
    CUDA_CHECK(cudaGetLastError());

    // Notice we pass h_points directly now, without .data()
    CUDA_CHECK(cudaMemcpy(h_points, d_points, n * sizeof(Point), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaFree(d_points));
    CUDA_CHECK(cudaFree(d_H_knn));
    CUDA_CHECK(cudaFree(d_Cmin));
    CUDA_CHECK(cudaFree(d_cdf_at_intensity));

    d_H_knn = nullptr;
    d_points = nullptr;

    knn_write_output();
    
    CUDA_CHECK(cudaFreeHost(h_points));
    h_points = nullptr;
}