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
extern int* d_cls_hist;
extern int* d_intensities;
extern int* d_cls_cnt;
extern int* d_point_cls;
extern int* d_intensities_knn;
extern int* d_H_knn;

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

__global__ void compute_cdf_kernel(
    const int* H, const int* intensities, int* Cmin, int* cdf_at_intensity, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    int base = i * 256;
    int running = 0;
    int cmin = INT_MAX;
    int target = intensities[i];
    int target_cdf = 0;

    for (int b = 0; b < 256; b++) {
        running += H[base + b];
        if (running > 0 && cmin == INT_MAX) cmin = running;
        if (b == target) target_cdf = running;
    }

    Cmin[i] = cmin;
    cdf_at_intensity[i] = target_cdf;
}

__global__ void compute_kmeans_terms_kernel(
    const int* cls_hist, const int* point_cls, const int* intensities,
    int* Cmin, int* cdf_at_intensity, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    int cls = point_cls[i];
    int base = cls * 256;
    int running = 0;
    int cmin = INT_MAX;
    int target = intensities[i];
    int target_cdf = 0;

    for (int b = 0; b < 256; b++) {
        running += cls_hist[base + b];
        if (running > 0 && cmin == INT_MAX) cmin = running;
        if (b == target) target_cdf = running;
    }

    Cmin[i] = cmin;
    cdf_at_intensity[i] = target_cdf;
}

__global__ void update_intensities_kernel(
    const int* cdf_at_intensity, const int* Cmin, int* intensities,
    const int* h_cls_cnt, const int* h_point_cls,
    int n, int k, bool is_kmeans)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    int cmin = Cmin[i];
    int m    = is_kmeans ? h_cls_cnt[h_point_cls[i]] : (k + 1);

    if (m != cmin) {
        int val = cdf_at_intensity[i];
        intensities[i] = max(0, (int)(((float)(val - cmin) / (float)(m - cmin))*255.0));
    }
}

__global__ void update_intensities_kmeans_kernel(
    const int* cdf_at_intensity, const int* Cmin, int* intensities,
    const int* d_cls_cnt, const int* d_point_cls, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    int cmin = Cmin[i];
    int m = d_cls_cnt[d_point_cls[i]];
    if (m != cmin) {
        intensities[i] = max(0, (int)((((float)(cdf_at_intensity[i] - cmin)) / (float)(m - cmin))*255.0));
    }
}

int main(int argc, char const* argv[]) {
    if (argc != 3) return 1;
    string inpfile = argv[1];
    model = argv[2];

    double st = omp_get_wtime();

    takeinput(inpfile);

    if (model == "kmeans") {
        kmeans();

        if (!d_cls_hist || !d_intensities || !d_cls_cnt || !d_point_cls) return 1;

        int *d_Cmin = nullptr, *d_cdf_at_intensity = nullptr;

        CUDA_CHECK(cudaMalloc(&d_Cmin,             n * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_cdf_at_intensity, n * sizeof(int)));
        int threads = 256;
        int blocks = (n + threads - 1) / threads;
        compute_kmeans_terms_kernel<<<blocks, threads>>>(d_cls_hist, d_point_cls, d_intensities, d_Cmin, d_cdf_at_intensity, n);
        CUDA_CHECK(cudaGetLastError());

        update_intensities_kmeans_kernel<<<(n + 255) / 256, 256>>>(d_cdf_at_intensity, d_Cmin, d_intensities,d_cls_cnt, d_point_cls, n);
        CUDA_CHECK(cudaGetLastError());

        CUDA_CHECK(cudaMemcpy(intensities.data(), d_intensities, n * sizeof(int), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaDeviceSynchronize());

        cudaFree(d_Cmin);
        cudaFree(d_cdf_at_intensity);
        cudaFree(d_cls_cnt);
        cudaFree(d_point_cls);
        cudaFree(d_cls_hist);
        cudaFree(d_intensities);
        d_cls_cnt = nullptr;
        d_point_cls = nullptr;
        d_cls_hist = nullptr;
        d_intensities = nullptr;
    } else if (model == "knn" || model == "approx_knn") {
        if (model == "knn") knn();
        else approx_knn();

        int *d_Cmin = nullptr, *d_cdf_at_intensity = nullptr;

        CUDA_CHECK(cudaMalloc(&d_Cmin,        n * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_cdf_at_intensity, n * sizeof(int)));

        int threads = 256;
        int blocks = (n + threads - 1) / threads;
        compute_cdf_kernel<<<blocks, threads>>>(d_H_knn, d_intensities_knn, d_Cmin, d_cdf_at_intensity, n);
        CUDA_CHECK(cudaGetLastError());

        update_intensities_kernel<<<(n + 255) / 256, 256>>>(
            d_cdf_at_intensity, d_Cmin, d_intensities_knn,
            nullptr, nullptr,
            n, k, false);
        CUDA_CHECK(cudaGetLastError());

        CUDA_CHECK(cudaMemcpy(intensities.data(), d_intensities_knn, n * sizeof(int), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaDeviceSynchronize());

        cudaFree(d_H_knn);
        cudaFree(d_Cmin);
        cudaFree(d_cdf_at_intensity);
        cudaFree(d_intensities_knn);
    } else {
        return 1;
    }

    genopt();

    cout << "gpu : " << omp_get_wtime() - st << " s" << endl;
    return 0;
}
