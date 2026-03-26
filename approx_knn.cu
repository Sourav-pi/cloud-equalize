#include "common.hpp"
#include <cuda_runtime.h>

static constexpr int THREADS_PER_BLOCK = 256;
static constexpr int KMAX = 128;
static constexpr int PQ_BINS = 32;

struct Neighbor {
    long long dist;
    int idx;
};

__device__ bool better_neighbor(const Neighbor& a, const Neighbor& b, const int* x, const int* y, const int* z) {
    if (a.dist != b.dist) return a.dist < b.dist;
    if (x[a.idx] != x[b.idx]) return x[a.idx] < x[b.idx];
    if (y[a.idx] != y[b.idx]) return y[a.idx] < y[b.idx];
    if (z[a.idx] != z[b.idx]) return z[a.idx] < z[b.idx];
    return a.idx < b.idx;
}

__device__ bool worse_neighbor(const Neighbor& a, const Neighbor& b, const int* x, const int* y, const int* z) {
    return better_neighbor(b, a, x, y, z);
}

__device__ void ivfpq_bubble_up(Neighbor* heap, int pos, const int* x, const int* y, const int* z) {
    while (pos > 0) {
        int parent = (pos - 1) / 2;
        if (!worse_neighbor(heap[pos], heap[parent], x, y, z)) break;
        Neighbor tmp = heap[parent];
        heap[parent] = heap[pos];
        heap[pos] = tmp;
        pos = parent;
    }
}

__device__ void ivfpq_bubble_down(Neighbor* heap, int size, int pos, const int* x, const int* y, const int* z) {
    while (true) {
        int left = 2 * pos + 1;
        int right = left + 1;
        int worst = pos;

        if (left < size && worse_neighbor(heap[left], heap[worst], x, y, z)) worst = left;
        if (right < size && worse_neighbor(heap[right], heap[worst], x, y, z)) worst = right;

        if (worst == pos) break;
        Neighbor tmp = heap[pos];
        heap[pos] = heap[worst];
        heap[worst] = tmp;
        pos = worst;
    }
}

__global__ void ivfpq_search_kernel(
    const int* x,
    const int* y,
    const int* z,
    const int* intens,
    const unsigned char* code_x,
    const unsigned char* code_y,
    const unsigned char* code_z,
    const int* cb_x,
    const int* cb_y,
    const int* cb_z,
    const int* candidates,
    const int* cand_count,
    int max_cands,
    int n,
    int k,
    int* H_flat)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    int eff_k = k;
    if (eff_k > n - 1) eff_k = n - 1;
    if (eff_k > KMAX) eff_k = KMAX;
    if (eff_k <= 0) return;

    Neighbor heap[KMAX];
    int heap_size = 0;

    long long qx = x[i];
    long long qy = y[i];
    long long qz = z[i];

    int count = cand_count[i];
    if (count > max_cands) count = max_cands;

    int base = i * max_cands;
    for (int t = 0; t < count; t++) {
        int idx = candidates[base + t];
        if (idx < 0 || idx == i) continue;

        long long dx = qx - x[idx];
        long long dy = qy - y[idx];
        long long dz = qz - z[idx];

        Neighbor cand;
        cand.dist = dx * dx + dy * dy + dz * dz;
        cand.idx = idx;

        if (heap_size < eff_k) {
            heap[heap_size] = cand;
            ivfpq_bubble_up(heap, heap_size, x, y, z);
            heap_size++;
        } else if (better_neighbor(cand, heap[0], x, y, z)) {
            heap[0] = cand;
            ivfpq_bubble_down(heap, heap_size, 0, x, y, z);
        }
    }

    int hrow = i * 256;
    for (int t = 0; t < heap_size; t++) {
        int idx = heap[t].idx;
        int bin = intens[idx];
        if (bin >= 0 && bin < 256) {
            H_flat[hrow + bin] += 1;
        }
    }
}

static inline int encode_scalar(int v, int mn, int mx) {
    if (mx <= mn) return 0;
    double r = (double)(v - mn) / (double)(mx - mn);
    int code = (int)floor(r * PQ_BINS);
    if (code < 0) code = 0;
    if (code >= PQ_BINS) code = PQ_BINS - 1;
    return code;
}

void approx_knn(vector<int>& H) {
    if (n <= 1 || k <= 0) {
        fill(H.begin(), H.end(), 0);
        return;
    }

    fill(H.begin(), H.end(), 0);

    // --------------------
    // 1) IVF indexing
    // --------------------
    int coarse_k = (int)sqrt((double)n);
    if (coarse_k < 8) coarse_k = 8;
    if (coarse_k > 64) coarse_k = 64;
    if (coarse_k > n) coarse_k = n;

    vector<int> cent_x(coarse_k), cent_y(coarse_k), cent_z(coarse_k);
    for (int c = 0; c < coarse_k; c++) {
        int idx = (int)((1LL * c * n) / coarse_k);
        if (idx >= n) idx = n - 1;
        cent_x[c] = h_x[idx];
        cent_y[c] = h_y[idx];
        cent_z[c] = h_z[idx];
    }

    vector<int> assign(n, 0);
    const int kmeans_iters = 10;
    for (int it = 0; it < kmeans_iters; it++) {
        vector<long long> sx(coarse_k, 0), sy(coarse_k, 0), sz(coarse_k, 0);
        vector<int> cnt(coarse_k, 0);

        for (int i = 0; i < n; i++) {
            long long best = LLONG_MAX;
            int best_c = 0;
            for (int c = 0; c < coarse_k; c++) {
                long long dx = (long long)h_x[i] - cent_x[c];
                long long dy = (long long)h_y[i] - cent_y[c];
                long long dz = (long long)h_z[i] - cent_z[c];
                long long d = dx * dx + dy * dy + dz * dz;
                if (d < best) {
                    best = d;
                    best_c = c;
                }
            }
            assign[i] = best_c;
            sx[best_c] += h_x[i];
            sy[best_c] += h_y[i];
            sz[best_c] += h_z[i];
            cnt[best_c]++;
        }

        for (int c = 0; c < coarse_k; c++) {
            if (cnt[c] > 0) {
                cent_x[c] = (int)(sx[c] / cnt[c]);
                cent_y[c] = (int)(sy[c] / cnt[c]);
                cent_z[c] = (int)(sz[c] / cnt[c]);
            }
        }
    }

    vector<vector<int> > lists(coarse_k);
    for (int i = 0; i < n; i++) {
        lists[assign[i]].push_back(i);
    }

    // --------------------
    // 2) PQ compression (3 subspaces: x,y,z)
    // --------------------
    int min_x = *min_element(h_x.begin(), h_x.end());
    int max_x = *max_element(h_x.begin(), h_x.end());
    int min_y = *min_element(h_y.begin(), h_y.end());
    int max_y = *max_element(h_y.begin(), h_y.end());
    int min_z = *min_element(h_z.begin(), h_z.end());
    int max_z = *max_element(h_z.begin(), h_z.end());

    vector<int> cb_x(PQ_BINS), cb_y(PQ_BINS), cb_z(PQ_BINS);
    for (int b = 0; b < PQ_BINS; b++) {
        cb_x[b] = min_x + (int)((2.0 * b + 1.0) * (max_x - min_x) / (2.0 * PQ_BINS));
        cb_y[b] = min_y + (int)((2.0 * b + 1.0) * (max_y - min_y) / (2.0 * PQ_BINS));
        cb_z[b] = min_z + (int)((2.0 * b + 1.0) * (max_z - min_z) / (2.0 * PQ_BINS));
    }

    vector<unsigned char> code_x(n), code_y(n), code_z(n);
    for (int i = 0; i < n; i++) {
        code_x[i] = (unsigned char)encode_scalar(h_x[i], min_x, max_x);
        code_y[i] = (unsigned char)encode_scalar(h_y[i], min_y, max_y);
        code_z[i] = (unsigned char)encode_scalar(h_z[i], min_z, max_z);
    }

    // --------------------
    // 3) Candidate selection (nprobe nearest coarse centroids)
    // --------------------
    int nprobe = k / 4 + 1;
    if (nprobe < 2) nprobe = 2;
    if (nprobe > 16) nprobe = 16;
    if (nprobe > coarse_k) nprobe = coarse_k;

    int max_cands = 16 * k;
    if (max_cands < 128) max_cands = 128;
    if (max_cands > 2048) max_cands = 2048;
    if (max_cands > n - 1) max_cands = n - 1;

    vector<int> candidates(1LL * n * max_cands, -1);
    vector<int> cand_count(n, 0);
    vector<int> seen(n, -1);

    for (int i = 0; i < n; i++) {
        vector<pair<long long, int> > cc;
        cc.reserve(coarse_k);

        for (int c = 0; c < coarse_k; c++) {
            long long dx = (long long)h_x[i] - cent_x[c];
            long long dy = (long long)h_y[i] - cent_y[c];
            long long dz = (long long)h_z[i] - cent_z[c];
            cc.push_back(make_pair(dx * dx + dy * dy + dz * dz, c));
        }

        nth_element(cc.begin(), cc.begin() + nprobe, cc.end());

        int out_base = i * max_cands;
        int cnt = 0;

        for (int t_idx = 0; t_idx < nprobe && cnt < max_cands; t_idx++) {
            int cid = cc[t_idx].second;
            const vector<int>& lst = lists[cid];
            for (int idx : lst) {
                if (idx == i) continue;
                if (seen[idx] == i) continue;
                seen[idx] = i;
                candidates[out_base + cnt] = idx;
                cnt++;
                if (cnt >= max_cands) break;
            }
        }

        // Fallback: pad with any unseen points.
        for (int j = 0; j < n && cnt < max_cands; j++) {
            if (j == i) continue;
            if (seen[j] == i) continue;
            seen[j] = i;
            candidates[out_base + cnt] = j;
            cnt++;
        }

        cand_count[i] = cnt;
    }

    // --------------------
    // 4) GPU parallel search + rerank
    // --------------------
    int *d_x = nullptr, *d_y = nullptr, *d_z = nullptr;
    int *d_intens = nullptr;
    int *d_candidates = nullptr;
    int *d_cand_count = nullptr;
    int *d_cb_x = nullptr, *d_cb_y = nullptr, *d_cb_z = nullptr;
    unsigned char *d_code_x = nullptr, *d_code_y = nullptr, *d_code_z = nullptr;
    int *d_H = nullptr;

    CUDA_CHECK(cudaMalloc(&d_x, n * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_y, n * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_z, n * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_intens, n * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_candidates, 1LL * n * max_cands * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_cand_count, n * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_cb_x, PQ_BINS * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_cb_y, PQ_BINS * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_cb_z, PQ_BINS * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_code_x, n * sizeof(unsigned char)));
    CUDA_CHECK(cudaMalloc(&d_code_y, n * sizeof(unsigned char)));
    CUDA_CHECK(cudaMalloc(&d_code_z, n * sizeof(unsigned char)));
    CUDA_CHECK(cudaMalloc(&d_H, 1LL * n * 256 * sizeof(int)));

    CUDA_CHECK(cudaMemcpy(d_x, h_x.data(), n * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_y, h_y.data(), n * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_z, h_z.data(), n * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_intens, intensities.data(), n * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_candidates, candidates.data(), 1LL * n * max_cands * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_cand_count, cand_count.data(), n * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_cb_x, cb_x.data(), PQ_BINS * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_cb_y, cb_y.data(), PQ_BINS * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_cb_z, cb_z.data(), PQ_BINS * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_code_x, code_x.data(), n * sizeof(unsigned char), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_code_y, code_y.data(), n * sizeof(unsigned char), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_code_z, code_z.data(), n * sizeof(unsigned char), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_H, 0, 1LL * n * 256 * sizeof(int)));

    dim3 block_dim(THREADS_PER_BLOCK);
    dim3 grid_dim((n + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK);

    ivfpq_search_kernel<<<grid_dim, block_dim>>>(
        d_x, d_y, d_z,
        d_intens,
        d_code_x, d_code_y, d_code_z,
        d_cb_x, d_cb_y, d_cb_z,
        d_candidates, d_cand_count,
        max_cands,
        n, k,
        d_H);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(H.data(), d_H, 1LL * n * 256 * sizeof(int), cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_x));
    CUDA_CHECK(cudaFree(d_y));
    CUDA_CHECK(cudaFree(d_z));
    CUDA_CHECK(cudaFree(d_intens));
    CUDA_CHECK(cudaFree(d_candidates));
    CUDA_CHECK(cudaFree(d_cand_count));
    CUDA_CHECK(cudaFree(d_cb_x));
    CUDA_CHECK(cudaFree(d_cb_y));
    CUDA_CHECK(cudaFree(d_cb_z));
    CUDA_CHECK(cudaFree(d_code_x));
    CUDA_CHECK(cudaFree(d_code_y));
    CUDA_CHECK(cudaFree(d_code_z));
    CUDA_CHECK(cudaFree(d_H));
}
