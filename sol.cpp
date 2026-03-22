#include <bits/stdc++.h>
#include <cuda_runtime.h>
using namespace std;

#define mt make_tuple

// static map<tuple<int,int,int>,int> point_to_idx;

static vector<int> point_x;
static vector<int> point_y;
static vector<int> point_z;
static vector<int> intensities;
static int n;
static int k;
static int t;
static string model;

__global__ void compute_distances_kernel(int* x, int* y, int* z, int* distsq, int n) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n && j < n) {
        int dx = x[i] - x[j];
        int dy = y[i] - y[j];
        int dz = z[i] - z[j];
        distsq[i * n + j] = dx * dx + dy * dy + dz * dz;
    }
}

void compute_distances(vector<vector<int>>& distsq) {
    int *d_x, *d_y, *d_z, *d_distsq;

    cudaMalloc(&d_x, n * sizeof(int));
    cudaMalloc(&d_y, n * sizeof(int));
    cudaMalloc(&d_z, n * sizeof(int));
    cudaMalloc(&d_distsq, n * n * sizeof(int));

    cudaMemcpy(d_x, point_x.data(), n * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, point_y.data(), n * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_z, point_z.data(), n * sizeof(int), cudaMemcpyHostToDevice);

    dim3 blockDim(16, 16);
    dim3 gridDim((n + blockDim.x - 1) / blockDim.x, (n + blockDim.y - 1) / blockDim.y);

    compute_distances_kernel<<<gridDim, blockDim>>>(d_x, d_y, d_z, d_distsq, n);
    int* h_distsq = (int*)malloc(n * n * sizeof(int));
    cudaMemcpy(h_distsq, d_distsq, n * n * sizeof(int), cudaMemcpyDeviceToHost);

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
    vector<vector<int>> distsq(n, vector<int>(n));

    compute_distances(distsq);
    for (int i = 0; i < n; i++) {
        vector<pair<int, int>> tmp;
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

void kmeans(vector<vector<int>>& H) {
    
}

void takeinput(string inpfile) {
    ifstream file(inpfile.c_str());
    file >> n >> k >> t;
    for (int i = 0; i < n; i++) {
        int x, y, z, I;
        file >> x >> y >> z >> I;
        point_x.push_back(x);
        point_y.push_back(y);
        point_z.push_back(z);
        intensities.push_back(I);
        // point_to_idx[mt(x,y,z)]=i;
    }
    file.close();
}

void genopt() {
    ofstream file((model + ".txt").c_str());
    for (int i = 0; i < n; i++) {
        file << point_x[i] << ' ' << point_y[i] << ' ' << point_z[i] << ' ' << intensities[i] << '\n';
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
    int m;
    if (model == "knn") m = k;
    for (int i = 0; i < n; i++) {
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
