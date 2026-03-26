#include "common.hpp"

vector<int> h_x;
vector<int> h_y;
vector<int> h_z;
vector<int> intensities;
int n;
int k;
int t;
string model;
extern vector<int> h_cls_cnt;
extern vector<int> h_point_cls;

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

void print_histograms(const vector<int>& H) {
    for (int i = 0; i < n; i++) {
        cout << "H[" << i << "] = { ";
        bool first = true;
        int total = 0;
        int base = i * 256;
        for (int j = 0; j < 256; j++) {
            int val = H[base + j];
            total += val;
            if (val == 0) continue;
            if (!first) cout << ", ";
            cout << j << ":" << val;
            first = false;
        }
        if (first) cout << "empty";
        cout << " } (total=" << total << ")\n";
    }
}

__global__ void compute_cdf_kernel(int* CDF, int* Cmin, int n) {
    __shared__ int s[256];
    int i = blockIdx.x;
    int j = threadIdx.x;
    if (i >= n) return;

    s[j] = CDF[i * 256 + j];
    __syncthreads();

    #pragma unroll
    for (int stride = 1; stride < 256; stride <<= 1) {
        int add = (j >= stride) ? s[j - stride] : 0;
        __syncthreads();
        s[j] += add;
        __syncthreads();
    }

    CDF[i * 256 + j] = s[j];

    if (j == 0) {
        int cmin = INT_MAX;
        for (int b = 0; b < 256; b++) {
            if (s[b] > 0) { cmin = s[b]; break; }
        }
        Cmin[i] = cmin;
    }
}

__global__ void update_intensities_kernel(
    const int* CDF, const int* Cmin, int* intensities,
    const int* h_cls_cnt, const int* h_point_cls,
    int n, int k, bool is_kmeans)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    int cmin = Cmin[i];
    int m    = is_kmeans ? h_cls_cnt[h_point_cls[i]] : (k + 1);

    if (m != cmin) {
        int val = CDF[i * 256 + intensities[i]];
        intensities[i] = max(0, (val - cmin) * 255 / (m - cmin));
    }
}

// ── main: only the two CPU calls swapped for GPU launches ───────────────────
int main(int argc, char const* argv[]) {
    if (argc != 3) return 1;
    string inpfile = argv[1];
    model = argv[2];

    double st = omp_get_wtime();

    takeinput(inpfile);

    vector<int> H(1LL * n * 256, 0);

    if      (model == "knn")       knn(H);
    else if (model == "approx_knn") approx_knn(H);
    else if (model == "kmeans")    kmeans(H);
    else return 1;

    int *d_CDF, *d_Cmin, *d_intensities;
    int *d_cls_cnt = nullptr, *d_point_cls = nullptr;

    CUDA_CHECK(cudaMalloc(&d_CDF,         1LL * n * 256 * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_Cmin,        n * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_intensities, n * sizeof(int)));

    CUDA_CHECK(cudaMemcpy(d_CDF,         H.data(),            1LL * n * 256 * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_intensities, intensities.data(),  n * sizeof(int),             cudaMemcpyHostToDevice));

    if (model == "kmeans") {
        CUDA_CHECK(cudaMalloc(&d_cls_cnt,   k * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_point_cls, n * sizeof(int)));
        CUDA_CHECK(cudaMemcpy(d_cls_cnt,   h_cls_cnt.data(),   k * sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_point_cls, h_point_cls.data(), n * sizeof(int), cudaMemcpyHostToDevice));
    }

    compute_cdf_kernel<<<n, 256>>>(d_CDF, d_Cmin, n);
    CUDA_CHECK(cudaGetLastError());

    update_intensities_kernel<<<(n + 255) / 256, 256>>>(
        d_CDF, d_Cmin, d_intensities,
        d_cls_cnt, d_point_cls,
        n, k, model == "kmeans");
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaMemcpy(intensities.data(), d_intensities, n * sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaFree(d_CDF); cudaFree(d_Cmin); cudaFree(d_intensities);
    if (d_cls_cnt)   cudaFree(d_cls_cnt);
    if (d_point_cls) cudaFree(d_point_cls);

    genopt();

    cout << "gpu : "<< omp_get_wtime() - st << " s" << endl;
    return 0;
}
