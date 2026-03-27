#include "common.hpp"
#include <cuda_runtime.h>
#include <cstdint>
#include <fstream>
#include <vector>
#include <algorithm>
#include <string>

using namespace std;

struct __align__(16) Point {
    int x;
    int y;
    int z;
    int intensity;
};

static Point* h_points = nullptr;
static int n;
static int k;
static int t;
static string model;
static constexpr int KMAX = 128;
static constexpr int THREADS_PER_BLOCK = 128;
static constexpr int WIN = 15000;
static constexpr int SHARED_BLOCKS = 3;

static void aknn_read_input(const string& inpfile) {
    ifstream file(inpfile.c_str());
    file >> n >> k >> t;
    
    CUDA_CHECK(cudaMallocHost(&h_points, n * sizeof(Point)));
    
    for (int i = 0; i < n; i++) {
        file >> h_points[i].x >> h_points[i].y >> h_points[i].z >> h_points[i].intensity;
    }
    file.close();
}

static void aknn_write_output() {
    ofstream file("approx_knn.txt");
    for (int i = 0; i < n; i++) {
        file << h_points[i].x << " " << h_points[i].y << " " << h_points[i].z << " " << h_points[i].intensity << "\n";
    }
    file.close();
}

// CPU Morton Calculation
static long long compute_morton(int x, int y, int z) {
    long long code = 0;
    x += 10000;
    y += 10000;
    z += 10000;
    for (int i = 0; i < 21; i++) {
        code |= ((long long)((x >> i) & 1u) << (3 * i));
        code |= ((long long)((y >> i) & 1u) << (3 * i + 1));
        code |= ((long long)((z >> i) & 1u) << (3 * i + 2));
    }
    return code;
}

// Gather kernel to arrange Points into Morton-sorted order on the GPU
__global__ void aknn_gather_sorted(const int* sidxs, const Point* in_points, Point* spoints, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        spoints[i] = in_points[sidxs[i]];
    }
}

// Lexicographical tie-breaking
__device__ __forceinline__ bool better_knn(int cand_dist, int cand_idx, int heap_dist, int heap_idx, const Point* spoints) {
    if (cand_dist != heap_dist) return cand_dist < heap_dist;
    if (spoints[cand_idx].x != spoints[heap_idx].x) return spoints[cand_idx].x < spoints[heap_idx].x;
    if (spoints[cand_idx].y != spoints[heap_idx].y) return spoints[cand_idx].y < spoints[heap_idx].y;
    return spoints[cand_idx].z < spoints[heap_idx].z;
}

__device__ __forceinline__ bool worse_knn(int a_dist, int a_idx, int b_dist, int b_idx, const Point* spoints) {
    if (a_dist != b_dist) return a_dist > b_dist;
    if (spoints[a_idx].x != spoints[b_idx].x) return spoints[a_idx].x > spoints[b_idx].x;
    if (spoints[a_idx].y != spoints[b_idx].y) return spoints[a_idx].y > spoints[b_idx].y;
    return spoints[a_idx].z > spoints[b_idx].z;
}

// SINGLE FUSED AKNN KERNEL WITH MULTI-BLOCK TILING
__global__ void aknn_fused_kernel(const int* sidxs, const Point* spoints, Point* out_points, int n, int k) {
    const int i = blockIdx.x * THREADS_PER_BLOCK + threadIdx.x;
    const bool is_valid = (i < n);
    const int eff_k = min(min(k, n - 1), KMAX);

    const int bc = (int)blockIdx.x * THREADS_PER_BLOCK + THREADS_PER_BLOCK / 2;
    const int hw = WIN / 2;
    const int win_st = max(0, bc - hw);
    const int win_ed = min(n - 1, win_st + WIN);

    Point target_pt;
    if (is_valid) {
        target_pt = spoints[i];
    }

    int hdist[KMAX];
    int hspos[KMAX];
    int hs = 0;

    // The user-requested enlarged tile: 
    // Example: 128 threads * (2*4 + 1) = 1152 elements per chunk
    constexpr int TILE_SIZE = THREADS_PER_BLOCK * (2 * SHARED_BLOCKS + 1);
    __shared__ Point tile[TILE_SIZE];

    // Build the Heap within the Local Window
    // Notice how we now advance the loop by TILE_SIZE instead of THREADS_PER_BLOCK
    for (int ts = win_st; ts <= win_ed; ts += TILE_SIZE) {
        
        // 1. Load the enlarged chunk cooperatively.
        // Because the chunk is larger than the thread count, threads must loop to fill it.
        const int tlen = min(TILE_SIZE, win_ed - ts + 1);
        for (int load_idx = threadIdx.x; load_idx < tlen; load_idx += THREADS_PER_BLOCK) {
            tile[load_idx] = spoints[ts + load_idx];
        }
        __syncthreads();

        // 2. Compute distances against the newly loaded large tile
        if (is_valid) {
            #pragma unroll 4
            for (int t = 0; t < tlen; t++) {
                const int j = ts + t;
                if (j == i) continue;

                const int dx = target_pt.x - tile[t].x;
                const int dy = target_pt.y - tile[t].y;
                const int dz = target_pt.z - tile[t].z;
                const int dist = dx * dx + dy * dy + dz * dz;

                if (hs < eff_k) {
                    hdist[hs] = dist;
                    hspos[hs] = j;
                    int p = hs++;
                    
                    while (p > 0) {
                        const int par = (p - 1) >> 1;
                        if (!worse_knn(hdist[p], hspos[p], hdist[par], hspos[par], spoints)) break;
                        int td = hdist[par]; hdist[par] = hdist[p]; hdist[p] = td;
                        int ti = hspos[par]; hspos[par] = hspos[p]; hspos[p] = ti;
                        p = par;
                    }
                } else if (hs > 0 && better_knn(dist, j, hdist[0], hspos[0], spoints)) {
                    hdist[0] = dist;
                    hspos[0] = j;
                    int p = 0;
                    while (true) {
                        const int l = 2 * p + 1, r = 2 * p + 2;
                        int mx = p;
                        if (l < hs && worse_knn(hdist[l], hspos[l], hdist[mx], hspos[mx], spoints)) mx = l;
                        if (r < hs && worse_knn(hdist[r], hspos[r], hdist[mx], hspos[mx], spoints)) mx = r;
                        if (mx == p) break;
                        int td = hdist[p]; hdist[p] = hdist[mx]; hdist[mx] = td;
                        int ti = hspos[p]; hspos[p] = hspos[mx]; hspos[mx] = ti;
                        p = mx;
                    }
                }
            }
        }
        __syncthreads();
    }

    // Mathematical Histogram Equalization
    if (is_valid) {
        int min_val = target_pt.intensity;
        for (int t = 0; t < hs; t++) {
            int nbr_intensity = spoints[hspos[t]].intensity;
            if (nbr_intensity < min_val) {
                min_val = nbr_intensity;
            }
        }

        int cmin = 0;
        int target_cdf = 0;

        if (target_pt.intensity == min_val) cmin++;
        target_cdf++;

        for (int t = 0; t < hs; t++) {
            int nbr_intensity = spoints[hspos[t]].intensity;
            if (nbr_intensity == min_val) cmin++;
            if (nbr_intensity <= target_pt.intensity) target_cdf++;
        }

        int m = k + 1;
        int new_intensity = target_pt.intensity;
        if (m != cmin) {
            int num = target_cdf - cmin;
            if (num < 0) num = 0;
            new_intensity = (num * 255) / (m - cmin);
        }

        Point out_p = target_pt;
        out_p.intensity = new_intensity;
        out_points[sidxs[i]] = out_p;
    }
}

void approx_knn(const string& inpfile) {
    aknn_read_input(inpfile);
    
    vector<pair<long long, int>> codes(n);
    #pragma omp parallel for
    for (int i = 0; i < n; i++) {
        codes[i] = {compute_morton(h_points[i].x, h_points[i].y, h_points[i].z), i};
    }
    
    sort(codes.begin(), codes.end());

    vector<int> sidxs(n);
    #pragma omp parallel for
    for (int i = 0; i < n; i++) {
        sidxs[i] = codes[i].second;
    }

    Point *d_in_points = nullptr;
    Point *d_spoints = nullptr;
    Point *d_out_points = nullptr;
    int *d_sidxs = nullptr;

    CUDA_CHECK(cudaMalloc(&d_in_points, n * sizeof(Point)));
    CUDA_CHECK(cudaMalloc(&d_spoints, n * sizeof(Point)));
    CUDA_CHECK(cudaMalloc(&d_out_points, n * sizeof(Point)));
    CUDA_CHECK(cudaMalloc(&d_sidxs, n * sizeof(int)));

    CUDA_CHECK(cudaMemcpy(d_in_points, h_points, n * sizeof(Point), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_sidxs, sidxs.data(), n * sizeof(int), cudaMemcpyHostToDevice));

    dim3 block(THREADS_PER_BLOCK);
    dim3 grid((n + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK);

    aknn_gather_sorted<<<grid, block>>>(d_sidxs, d_in_points, d_spoints, n);
    CUDA_CHECK(cudaGetLastError());

    aknn_fused_kernel<<<grid, block>>>(d_sidxs, d_spoints, d_out_points, n, k);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(h_points, d_out_points, n * sizeof(Point), cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_in_points));
    CUDA_CHECK(cudaFree(d_spoints));
    CUDA_CHECK(cudaFree(d_out_points));
    CUDA_CHECK(cudaFree(d_sidxs));

    aknn_write_output();
    
    CUDA_CHECK(cudaFreeHost(h_points));
    h_points = nullptr;
}