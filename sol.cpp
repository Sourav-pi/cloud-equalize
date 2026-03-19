#include<bits/stdc++.h>
#include<cuda_runtime.h>
using namespace std;

#define mt make_tuple

static map<tuple<int,int,int>,int> point_to_idx;

static vector<int> point_x;
static vector<int> point_y;
static vector<int> point_z;
static vector<int> intensities;
static int n;
static int k;
static int t;

__global__ void compute_distances_kernel(int *x, int *y, int *z, int *distsq, int n) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (i < n && j < n) {
        int dx = x[i] - x[j];
        int dy = y[i] - y[j];
        int dz = z[i] - z[j];
        distsq[i * n + j] = dx*dx + dy*dy + dz*dz;
    }
}

void compute_pairwise_distances_cuda(vector<vector<int>>& distsq) {
    int *d_x, *d_y, *d_z, *d_distsq;
    
    cudaMalloc(&d_x, n * sizeof(int));
    cudaMalloc(&d_y, n * sizeof(int));
    cudaMalloc(&d_z, n * sizeof(int));
    cudaMalloc(&d_distsq, n * n * sizeof(int));
    
    cudaMemcpy(d_x, px.data(), n * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, py.data(), n * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_z, pz.data(), n * sizeof(int), cudaMemcpyHostToDevice);
    
    dim3 blockDim(16, 16);
    dim3 gridDim((n + blockDim.x - 1) / blockDim.x, (n + blockDim.y - 1) / blockDim.y);
    
    compute_distances_kernel<<<gridDim, blockDim>>>(d_x, d_y, d_z, d_distsq, n);
    int *h_distsq = (int*)malloc(n * n * sizeof(int));
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


void knn() {
    vector<vector<int>> H(n,vector<int>(256));
    vector<vector<int>> distsq(n,vector<int>(n));
    
    compute_pairwise_distances_cuda(distsq);
}

void approx_knn() {

}

void kmeans() {

}

void takeinput(string inpfile) {
    ifstream file(inpfile);
    file >> n >> k >> t;
    for(int i=0;i<n;i++){
        int x,y,z,I;
        file >> x >> y >> z >> I;
        point_x.push_back(x);
        point_y.push_back(y);
        point_z.push_back(z);
        intensities.push_back(I);
        point_to_idx[mt(x,y,z)]=i;
    }
    file.close();
}

int main(int argc, char const *argv[]) {
    if(argc != 2) return 1;
    string inpfile = argv[0];
    string model = argv[1];
    if(model=="knn") knn();
    else if(model=="approx_knn") approx_knn();
    else if(model=="kmeans") kmeans();
    else return 1;
    return 0;
}
