#ifndef COMMON_HPP
#define COMMON_HPP

#include <bits/stdc++.h>
#include <omp.h>
#include <cuda_runtime.h>

using namespace std;



void knn(const string& inpfile);
void approx_knn(const string& inpfile);
void kmeans(const string& inpfile);
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
