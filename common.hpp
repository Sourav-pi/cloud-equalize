#ifndef COMMON_HPP
#define COMMON_HPP

#include <bits/stdc++.h>
#include <omp.h>
#include <cuda_runtime.h>

using namespace std;

extern vector<int> h_x;
extern vector<int> h_y;
extern vector<int> h_z;
extern vector<int> intensities;
extern int n;
extern int k;
extern int t;
extern string model;

void knn(vector<int>& H);
void approx_knn(vector<int>& H);
void kmeans();
void print_histograms(const vector<int>& H);

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err__ = (call); \
        if (err__ != cudaSuccess) { \
            cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ \
                 << " in " << #call << ": " << cudaGetErrorString(err__) << "\n"; \
            exit(1); \
        } \
    } while (0)


#endif
