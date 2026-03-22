#include <bits/stdc++.h>
#include <cuda_runtime.h>
using namespace std;

#define mt make_tuple

// static map<tuple<int,int,int>,int> point_to_idx;

static vector<int> h_x;
static vector<int> h_y;
static vector<int> h_z;
static vector<int> intensities;
static int n;
static int k;
static int t;
static string model;

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

void compute_distances(vector<vector<long long>>& distsq) {
    int *d_x, *d_y, *d_z;
    long long *d_distsq;

    cudaMalloc(&d_x, n * sizeof(int));
    cudaMalloc(&d_y, n * sizeof(int));
    cudaMalloc(&d_z, n * sizeof(int));
    cudaMalloc(&d_distsq, n * n * sizeof(long long));

    cudaMemcpy(d_x, h_x.data(), n * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, h_y.data(), n * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_z, h_z.data(), n * sizeof(int), cudaMemcpyHostToDevice);

    dim3 blockDim(16, 16);
    dim3 gridDim((n + blockDim.x - 1) / blockDim.x, (n + blockDim.y - 1) / blockDim.y);

    compute_distances_kernel<<<gridDim, blockDim>>>(d_x, d_y, d_z, d_distsq, n);
    long long* h_distsq = (long long*)malloc(n * n * sizeof(long long));
    cudaMemcpy(h_distsq, d_distsq, n * n * sizeof(long long), cudaMemcpyDeviceToHost);

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            distsq[i][j] = h_distsq[i * n + j];
        }
    }

    cudaFree(d_x);
    cudaFree(d_y);
    cudaFree(d_z);
    cudaFree(d_distsq);
    free(h_distsq);
}

void knn(vector<vector<int>>& H) {
    vector<vector<long long>> distsq(n, vector<long long>(n));

    compute_distances(distsq);
    for (int i = 0; i < n; i++) {
        vector<pair<long long, int>> tmp;
        for (int j = 0; j < n; j++) {
            if (j != i) {
                tmp.push_back(make_pair(distsq[i][j], j));
            }
        }
        sort(tmp.begin(), tmp.end());
        for (int j = 0; j < k; j++) {
            H[i][intensities[tmp[j].second]] += 1;
        }
    }
}

void approx_knn(vector<vector<int>>& H) {

}


static vector<int> h_point_cls;
static vector<int> h_cls_cnt;
static vector<int> h_cls_md_x;
static vector<int> h_cls_md_y;
static vector<int> h_cls_md_z;

static int* d_km_x = nullptr;
static int* d_km_y = nullptr;
static int* d_km_z = nullptr;
static int* d_km_point_cls = nullptr;
static int* d_km_cls_cnt = nullptr;
static long long* d_km_sum_x = nullptr;
static long long* d_km_sum_y = nullptr;
static long long* d_km_sum_z = nullptr;
static int* d_km_accum_cnt = nullptr;

constexpr int MAX_K = 128;
__constant__ int c_cls_md_x[MAX_K];
__constant__ int c_cls_md_y[MAX_K];
__constant__ int c_cls_md_z[MAX_K];

__global__ void reassign_kernel(int* d_point_cls, int* d_cls_cnt, const int* d_x, const int* d_y, const int* d_z, int n, int k) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        long long minn = LLONG_MAX;
        int min_idx = 0;
        for (int j = 0; j < k; j++) {
            long long dx = (long long)c_cls_md_x[j] - d_x[i];
            long long dy = (long long)c_cls_md_y[j] - d_y[i];
            long long dz = (long long)c_cls_md_z[j] - d_z[i];
            long long sq = dx * dx + dy * dy + dz * dz;
            if (sq < minn) {
                minn = sq;
                min_idx = j;
            }
        }
        d_point_cls[i] = min_idx;
        atomicAdd(&d_cls_cnt[min_idx],1);
    }
}

void init_kmeans_device_buffers() {
    cudaMalloc(&d_km_x, n * sizeof(int));
    cudaMalloc(&d_km_y, n * sizeof(int));
    cudaMalloc(&d_km_z, n * sizeof(int));
    cudaMalloc(&d_km_point_cls, n * sizeof(int));
    cudaMalloc(&d_km_cls_cnt, k * sizeof(int));
    cudaMalloc(&d_km_sum_x, k * sizeof(long long));
    cudaMalloc(&d_km_sum_y, k * sizeof(long long));
    cudaMalloc(&d_km_sum_z, k * sizeof(long long));
    cudaMalloc(&d_km_accum_cnt, k * sizeof(int));

    cudaMemcpy(d_km_x, h_x.data(), n * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_km_y, h_y.data(), n * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_km_z, h_z.data(), n * sizeof(int), cudaMemcpyHostToDevice);
}

void free_kmeans_device_buffers() {
    cudaFree(d_km_x);
    cudaFree(d_km_y);
    cudaFree(d_km_z);
    cudaFree(d_km_point_cls);
    cudaFree(d_km_cls_cnt);
    cudaFree(d_km_sum_x);
    cudaFree(d_km_sum_y);
    cudaFree(d_km_sum_z);
    cudaFree(d_km_accum_cnt);

    d_km_x = nullptr;
    d_km_y = nullptr;
    d_km_z = nullptr;
    d_km_point_cls = nullptr;
    d_km_cls_cnt = nullptr;
    d_km_sum_x = nullptr;
    d_km_sum_y = nullptr;
    d_km_sum_z = nullptr;
    d_km_accum_cnt = nullptr;
}

void reassign_cls(bool copy_point_cls_to_host){
    int threads_per_block = 256;
    int blocks = (n + threads_per_block - 1) / threads_per_block;
    cudaMemset(d_km_cls_cnt, 0, k * sizeof(int));
    cudaMemcpyToSymbol(c_cls_md_x, h_cls_md_x.data(), k * sizeof(int));
    cudaMemcpyToSymbol(c_cls_md_y, h_cls_md_y.data(), k * sizeof(int));
    cudaMemcpyToSymbol(c_cls_md_z, h_cls_md_z.data(), k * sizeof(int));

    reassign_kernel<<<blocks, threads_per_block>>>(d_km_point_cls, d_km_cls_cnt, d_km_x, d_km_y, d_km_z, n, k);

    if (copy_point_cls_to_host) {
        cudaMemcpy(h_point_cls.data(), d_km_point_cls, n * sizeof(int), cudaMemcpyDeviceToHost);
    }
    cudaMemcpy(h_cls_cnt.data(), d_km_cls_cnt, k * sizeof(int), cudaMemcpyDeviceToHost);
}

__global__ void add_kernel(
    const int* d_x,
    const int* d_y,
    const int* d_z,
    const int* d_point_cls,
    long long* d_sum_x,
    long long* d_sum_y,
    long long* d_sum_z,
    int* d_cnt,
    int n,
    int k
) {
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
            atomicAdd(&s_sum_x[cls], (long long)d_x[i]);
            atomicAdd(&s_sum_y[cls], (long long)d_y[i]);
            atomicAdd(&s_sum_z[cls], (long long)d_z[i]);
            atomicAdd(&s_cnt[cls], 1);
        }
    }

    __syncthreads();
    for (int c = threadIdx.x; c < k; c += blockDim.x) {
        atomicAdd(&d_sum_x[c], s_sum_x[c]);
        atomicAdd(&d_sum_y[c], s_sum_y[c]);
        atomicAdd(&d_sum_z[c], s_sum_z[c]);
        atomicAdd(&d_cnt[c], s_cnt[c]);
    }
}

void recalculate_md() {
    int threads_per_block = 256;
    int blocks = (n + threads_per_block - 1) / threads_per_block;
    cudaMemset(d_km_sum_x, 0, k * sizeof(long long));
    cudaMemset(d_km_sum_y, 0, k * sizeof(long long));
    cudaMemset(d_km_sum_z, 0, k * sizeof(long long));
    cudaMemset(d_km_accum_cnt, 0, k * sizeof(int));

    add_kernel<<<blocks, threads_per_block>>>(
        d_km_x, d_km_y, d_km_z, d_km_point_cls, d_km_sum_x, d_km_sum_y, d_km_sum_z, d_km_accum_cnt, n, k
    );

    vector<long long> h_sum_x(k, 0), h_sum_y(k, 0), h_sum_z(k, 0);
    vector<int> h_new_cnt(k, 0);
    cudaMemcpy(h_sum_x.data(), d_km_sum_x, k * sizeof(long long), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_sum_y.data(), d_km_sum_y, k * sizeof(long long), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_sum_z.data(), d_km_sum_z, k * sizeof(long long), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_new_cnt.data(), d_km_accum_cnt, k * sizeof(int), cudaMemcpyDeviceToHost);
    h_cls_cnt = h_new_cnt;

    for (int i = 0; i < k; i++) {
        if (h_cls_cnt[i] > 0) {
            h_cls_md_x[i] = (int)(h_sum_x[i] / h_cls_cnt[i]);
            h_cls_md_y[i] = (int)(h_sum_y[i] / h_cls_cnt[i]);
            h_cls_md_z[i] = (int)(h_sum_z[i] / h_cls_cnt[i]);
        }
    }

}


void kmeans(vector<vector<int>>& H) {
    if (k <= 0 || k > MAX_K || n <= 0) return;

    h_point_cls.resize(n);
    h_cls_cnt.resize(k);
    h_cls_md_x.resize(k);
    h_cls_md_y.resize(k);
    h_cls_md_z.resize(k);
    // initial mds are the first k points
    for(int i=0;i<k;i++){
        h_cls_md_x[i]=h_x[i];
        h_cls_md_y[i]=h_y[i];
        h_cls_md_z[i]=h_z[i];
    }
    init_kmeans_device_buffers();
    for (int iter = 0; iter < t; iter++) {
        reassign_cls(false);
        recalculate_md();
    }
    reassign_cls(true);

    vector<vector<int>> cls_hist(k, vector<int>(256, 0));
    for (int i = 0; i < n; i++) {
        int cls = h_point_cls[i];
        if (cls >= 0 && cls < k) {
            cls_hist[cls][intensities[i]] += 1;
        }
    }
    for (int i = 0; i < n; i++) {
        int cls = h_point_cls[i];
        if (cls >= 0 && cls < k) {
            H[i] = cls_hist[cls];
        }
    }

    free_kmeans_device_buffers();

}

void takeinput(string inpfile) {
    ifstream file(inpfile.c_str());
    file >> n >> k >> t;
    for (int i = 0; i < n; i++) {
        int x, y, z, I;
        file >> x >> y >> z >> I;
        h_x.push_back(x);
        h_y.push_back(y);
        h_z.push_back(z);
        intensities.push_back(I);
        // point_to_idx[mt(x,y,z)]=i;
    }
    file.close();
}

void genopt() {
    ofstream file((model + ".txt").c_str());
    for (int i = 0; i < n; i++) {
        file << h_x[i] << ' ' << h_y[i] << ' ' << h_z[i] << ' ' << intensities[i] << '\n';
    }
    file.close();
}

void compute_cdf(vector<vector<int> >& CDF, vector<int>& Cmin) {
    for (int i = 0; i < n; i++) {
        if(CDF[i][0]>0) Cmin[i]=CDF[i][0];
        for (int j = 1; j < 256; j++) {
            CDF[i][j] += CDF[i][j - 1];
            if (CDF[i][j] > 0) {
                Cmin[i] = min(Cmin[i], CDF[i][j]);
            }
        }
    }
}

void print_histograms(const vector<vector<int>>& H) {
    for (int i = 0; i < n; i++) {
        cout << "H[" << i << "] = { ";
        bool first = true;
        int total = 0;
        for (int j = 0; j < 256; j++) {
            total += H[i][j];
            if (H[i][j] == 0) continue;
            if (!first) cout << ", ";
            cout << j << ":" << H[i][j];
            first = false;
        }
        if (first) cout << "empty";
        cout << " } (total=" << total << ")\n";
    }
}

void update_intensities(const vector<vector<int>>& CDF, const vector<int>& Cmin) {
    for (int i = 0; i < n; i++) {
        int m = (model == "knn") ? (k + 1) : CDF[i][255];
        if (m != Cmin[i])
            intensities[i] = max(0,(CDF[i][intensities[i]] - Cmin[i])) * 255 / (m - Cmin[i]);
    }
}

int main(int argc, char const* argv[]) {
    if (argc != 3) return 1;
    string inpfile = argv[1];
    model = argv[2];

    takeinput(inpfile);

    vector<vector<int>> H(n, vector<int>(256, 0));

    // generate hist
    if (model == "knn") knn(H);
    else if (model == "approx_knn") approx_knn(H);
    else if (model == "kmeans") kmeans(H);
    else return 1;

    // get CDF & Cmin
    auto CDF = H;
    vector<int> Cmin(n, INT_MAX);
    compute_cdf(CDF, Cmin);

    // update intensities
    update_intensities(CDF, Cmin); 

    genopt();

    cout << "Done........." << endl;
    return 0;
}
