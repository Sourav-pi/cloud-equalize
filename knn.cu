#include "common.hpp"
#include <cuda_runtime.h>
#include <fstream>
#include <string>

using namespace std;

struct __align__(16) Point {
    int x;
    int y;
    int z;
    int intensity;
};

// Replaced vector with a raw pointer for pinned memory
static Point* h_points = nullptr;
static int n;
static int k;
static int t;
static string model;
static constexpr int KMAX = 128;
static constexpr int THREADS_PER_BLOCK = 256;

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

// UPDATED: Lexicographical tie-breaking (Dist -> X -> Y -> Z)
__device__ __forceinline__ bool better_knn(int cand_dist, int cand_idx, int heap_dist, int heap_idx, const Point* points) {
    if (cand_dist != heap_dist) return cand_dist < heap_dist;
    if (points[cand_idx].x != points[heap_idx].x) return points[cand_idx].x < points[heap_idx].x;
    if (points[cand_idx].y != points[heap_idx].y) return points[cand_idx].y < points[heap_idx].y;
    return points[cand_idx].z < points[heap_idx].z;
}

// UPDATED: Lexicographical tie-breaking (Dist -> X -> Y -> Z)
__device__ __forceinline__ bool worse_knn(int a_dist, int a_idx, int b_dist, int b_idx, const Point* points) {
    if (a_dist != b_dist) return a_dist > b_dist;
    if (points[a_idx].x != points[b_idx].x) return points[a_idx].x > points[b_idx].x;
    if (points[a_idx].y != points[b_idx].y) return points[a_idx].y > points[b_idx].y;
    return points[a_idx].z > points[b_idx].z;
}

// SINGLE FUSED KERNEL
__global__ void knn_fused_kernel(const Point* in_points, Point* out_points, int n, int k) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Allocate shared memory for the tile
    __shared__ Point shared_points[THREADS_PER_BLOCK];

    int eff_k = k < (n - 1) ? k : (n - 1);

    int heap_dist[KMAX];
    int heap_idx[KMAX];
    int heap_size = 0;

    // Load the target point for this specific thread into registers
    int xi = 0, yi = 0, zi = 0;
    int target_intensity = 0; // Adjusted to match 'int' in struct
    
    if (i < n) {
        xi = in_points[i].x;
        yi = in_points[i].y;
        zi = in_points[i].z;
        target_intensity = in_points[i].intensity;
    }

    // Calculate how many tiles we need to process all 'n' points
    int num_tiles = (n + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    
    #pragma unroll 4
    for (int tile = 0; tile < num_tiles; ++tile) {
        // 1. Load data into shared memory
        int global_idx = tile * THREADS_PER_BLOCK + threadIdx.x;
        if (global_idx < n) {
            shared_points[threadIdx.x] = in_points[global_idx];
        }
        
        __syncthreads();

        // Only compute if this thread represents a valid target point
        if (i < n) {
            // 2. Compute distances against the tile
            int num_elements_in_tile = (tile == num_tiles - 1) ? (n - tile * THREADS_PER_BLOCK) : THREADS_PER_BLOCK;
            
            #pragma unroll
            for (int j = 0; j < num_elements_in_tile; ++j) {
                int cand_idx = tile * THREADS_PER_BLOCK + j;
                if (cand_idx == i) continue;

                int dx = xi - shared_points[j].x;
                int dy = yi - shared_points[j].y;
                int dz = zi - shared_points[j].z;
                int cand_dist = dx * dx + dy * dy + dz * dz;

                // Max-Heap logic (UPDATED TO PASS in_points FOR TIE-BREAKING)
                if (heap_size < eff_k) {
                    int p = heap_size;
                    heap_dist[p] = cand_dist;
                    heap_idx[p] = cand_idx;
                    heap_size++;

                    while (p > 0) {
                        int par = (p - 1) >> 1;
                        if (!worse_knn(heap_dist[p], heap_idx[p], heap_dist[par], heap_idx[par], in_points)) break;
                        int td = heap_dist[par]; heap_dist[par] = heap_dist[p]; heap_dist[p] = td;
                        int ti = heap_idx[par]; heap_idx[par] = heap_idx[p]; heap_idx[p] = ti;
                        p = par;
                    }
                } else if (eff_k > 0 && better_knn(cand_dist, cand_idx, heap_dist[0], heap_idx[0], in_points)) {
                    heap_dist[0] = cand_dist;
                    heap_idx[0] = cand_idx;

                    int p = 0;
                    while (true) {
                        int l = 2 * p + 1;
                        int r = l + 1;
                        int worst = p;

                        if (l < heap_size && worse_knn(heap_dist[l], heap_idx[l], heap_dist[worst], heap_idx[worst], in_points)) worst = l;
                        if (r < heap_size && worse_knn(heap_dist[r], heap_idx[r], heap_dist[worst], heap_idx[worst], in_points)) worst = r;

                        if (worst == p) break;
                        int td = heap_dist[p]; heap_dist[p] = heap_dist[worst]; heap_dist[worst] = td;
                        int ti = heap_idx[p]; heap_idx[p] = heap_idx[worst]; heap_idx[worst] = ti;
                        p = worst;
                    }
                }
            }
        }
        
        __syncthreads();
    }

    // 3. Mathematical Histogram Equalization (No 256-bin histogram required)
    if (i < n) {
        // Pass A: Find the minimum intensity in the K-neighborhood
        int min_val = target_intensity;
        for (int t = 0; t < heap_size; t++) {
            int nbr_intensity = in_points[heap_idx[t]].intensity;
            if (nbr_intensity < min_val) {
                min_val = nbr_intensity;
            }
        }

        // Pass B: Compute Cmin (count of min_val) and target_cdf (count <= target_intensity)
        int cmin = 0;
        int target_cdf = 0;

        // Account for the target point itself
        if (target_intensity == min_val) cmin++;
        target_cdf++; // target_intensity <= target_intensity is always true

        // Account for the neighbors
        for (int t = 0; t < heap_size; t++) {
            int nbr_intensity = in_points[heap_idx[t]].intensity;
            if (nbr_intensity == min_val) cmin++;
            if (nbr_intensity <= target_intensity) target_cdf++;
        }

        // 4. Update Equation
        int m = k + 1;
        int new_intensity = target_intensity; // Adjusted to match 'int'
        if (m != cmin) {
            int num = target_cdf - cmin;
            if (num < 0) num = 0;
            new_intensity = (num * 255) / (m - cmin);
        }

        // 5. Write out fully formed point to prevent Read-After-Write hazards
        Point out_p;
        out_p.x = xi;
        out_p.y = yi;
        out_p.z = zi;
        out_p.intensity = new_intensity;
        
        out_points[i] = out_p;
    }
}

void knn(const string& inpfile) {
    knn_read_input(inpfile);
    
    Point *d_in_points = nullptr;
    Point *d_out_points = nullptr;

    CUDA_CHECK(cudaMalloc(&d_in_points, n * sizeof(Point)));
    CUDA_CHECK(cudaMalloc(&d_out_points, n * sizeof(Point)));

    CUDA_CHECK(cudaMemcpy(d_in_points, h_points, n * sizeof(Point), cudaMemcpyHostToDevice));

    dim3 block_dim(THREADS_PER_BLOCK);
    dim3 grid_dim((n + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK);

    // Launch single fused kernel
    knn_fused_kernel<<<grid_dim, block_dim>>>(d_in_points, d_out_points, n, k);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // Copy back from the separate output array
    CUDA_CHECK(cudaMemcpy(h_points, d_out_points, n * sizeof(Point), cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_in_points));
    CUDA_CHECK(cudaFree(d_out_points));

    knn_write_output();
    
    CUDA_CHECK(cudaFreeHost(h_points));
    h_points = nullptr;
}