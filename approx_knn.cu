#include "common.hpp"
#include "knn_heap_utils.cuh"
#include <cuda_runtime.h>
#include <cstdint>

static constexpr int KMAX     = 128;
static constexpr int THREADS_PER_BLOCK      = 128;
static constexpr int WIN_MULT = 100;    // window = WIN_MULT * k candidates

int* d_intensities_aknn;
int* d_H_aknn;

static long long compute_morton(int x, int y, int z)
{
    long long code = 0;
    x += 10000; y += 10000; z += 10000;
    for (int i = 0; i < 21; i++) {
        code |= ((long long)((x >> i) & 1u) << (3*i));
        code |= ((long long)((y >> i) & 1u) << (3*i + 1));
        code |= ((long long)((z >> i) & 1u) << (3*i + 2));
    }
    return code;
}

__global__ void gather_sorted(
    const int* __restrict__ sidxs,
    const int* __restrict__ x,  const int* __restrict__ y,
    const int* __restrict__ z,  const int* __restrict__ intens,
    int* __restrict__ sx, int* __restrict__ sy,
    int* __restrict__ sz, int* __restrict__ si, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    int orig = sidxs[i];
    sx[i] = x[orig];
    sy[i] = y[orig];
    sz[i] = z[orig];
    si[i] = intens[orig];
}

__global__ void build_hist(
    const int* __restrict__ sidxs,   // sorted_idxs[spos] → original index
    const int* __restrict__ sx,      // sorted coords (sequential reads!)
    const int* __restrict__ sy,
    const int* __restrict__ sz,
    const int* __restrict__ si,      // sorted intensities
    int n, int k, int* __restrict__ H)
{
    const int i = blockIdx.x * THREADS_PER_BLOCK + threadIdx.x;
    if (i >= n) return;

    const int eff_k = min(k, n - 1);

    const int bc     = (int)blockIdx.x * THREADS_PER_BLOCK + THREADS_PER_BLOCK / 2;
    const int hw     = WIN_MULT * k / 2;
    const int win_st = max(0,     bc - hw);
    const int win_ed = min(n - 1, win_st + WIN_MULT * k);

    const int xi = sx[i], yi = sy[i], zi = sz[i];

    int hdist[KMAX];
    int hspos[KMAX];
    int hs = 0;

    __shared__ int tx[THREADS_PER_BLOCK], ty[THREADS_PER_BLOCK], tz[THREADS_PER_BLOCK];

    for (int ts = win_st; ts <= win_ed; ts += THREADS_PER_BLOCK) {

        const int lp = ts + threadIdx.x;
        if (lp <= win_ed) {
            tx[threadIdx.x] = sx[lp];
            ty[threadIdx.x] = sy[lp];
            tz[threadIdx.x] = sz[lp];
        }
        __syncthreads();

        const int tlen = min(THREADS_PER_BLOCK, win_ed - ts + 1);

        #pragma unroll 4
        for (int t = 0; t < tlen; t++) {
            const int j = ts + t;
            if (j == i) continue;

            const int dx   = xi - tx[t];
            const int dy   = yi - ty[t];
            const int dz   = zi - tz[t];
            const int dist = dx*dx + dy*dy + dz*dz;

            if (hs < eff_k) {
                hdist[hs] = dist;
                hspos[hs] = j;
                int p = hs++;
                while (p > 0) {
                    const int par = (p - 1) >> 1;
                    if (hdist[par] < hdist[p]) {
                        int td = hdist[par]; hdist[par] = hdist[p]; hdist[p] = td;
                        int ti = hspos[par]; hspos[par] = hspos[p]; hspos[p] = ti;
                        p = par;
                    } else break;
                }
            } else if (hs > 0 && dist < hdist[0]) {
                hdist[0] = dist;
                hspos[0] = j;
                int p = 0;
                while (true) {
                    const int l = 2*p+1, r = 2*p+2;
                    int mx = p;
                    if (l < hs && hdist[l] > hdist[mx]) mx = l;
                    if (r < hs && hdist[r] > hdist[mx]) mx = r;
                    if (mx == p) break;
                    int td = hdist[p]; hdist[p] = hdist[mx]; hdist[mx] = td;
                    int ti = hspos[p]; hspos[p] = hspos[mx]; hspos[mx] = ti;
                    p = mx;
                }
            }
        }
        __syncthreads();
    }

    int* hrow = H + (long long)sidxs[i] * 256;
    hrow[si[i]] += 1;                   
    for (int t = 0; t < hs; t++)
        hrow[si[hspos[t]]] += 1;
}

void approx_knn()
{
    int *d_x=nullptr, *d_y=nullptr, *d_z=nullptr;
    int *d_sx=nullptr, *d_sy=nullptr, *d_sz=nullptr, *d_si=nullptr;
    int *d_sidxs=nullptr;
    d_intensities_aknn = nullptr;
    d_H_aknn = nullptr;

    vector<pair<long long,int>> codes(n);
    #pragma omp parallel for
    for (int i = 0; i < n; i++)
        codes[i] = { compute_morton(h_x[i], h_y[i], h_z[i]), i };
    sort(codes.begin(), codes.end());

    vector<int> sidxs(n);
    #pragma omp parallel for
    for (int i = 0; i < n; i++) sidxs[i] = codes[i].second;

    CUDA_CHECK(cudaMalloc(&d_x,               n * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_y,               n * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_z,               n * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_intensities_aknn, n * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_sidxs,           n * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_sx,              n * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_sy,              n * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_sz,              n * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_si,              n * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_H_aknn,  1LL * n * 256 * sizeof(int)));

    CUDA_CHECK(cudaMemcpy(d_x,    h_x.data(),        n*sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_y,    h_y.data(),        n*sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_z,    h_z.data(),        n*sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_intensities_aknn, intensities.data(), n*sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_sidxs, sidxs.data(),     n*sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_H_aknn, 0, 1LL * n * 256 * sizeof(int)));

    dim3 block(THREADS_PER_BLOCK);
    dim3 grid((n + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK);

    gather_sorted<<<grid, block>>>(d_sidxs, d_x, d_y, d_z, d_intensities_aknn, d_sx, d_sy, d_sz, d_si, n);
    CUDA_CHECK(cudaGetLastError());

    build_hist<<<grid, block>>>(d_sidxs, d_sx, d_sy, d_sz, d_si, n, k, d_H_aknn);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaFree(d_x));
    CUDA_CHECK(cudaFree(d_y));
    CUDA_CHECK(cudaFree(d_z));
    CUDA_CHECK(cudaFree(d_sidxs));
    CUDA_CHECK(cudaFree(d_sx));
    CUDA_CHECK(cudaFree(d_sy));
    CUDA_CHECK(cudaFree(d_sz));
    CUDA_CHECK(cudaFree(d_si));
}