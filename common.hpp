#ifndef COMMON_HPP
#define COMMON_HPP

#include <bits/stdc++.h>

using namespace std;

extern vector<int> h_x;
extern vector<int> h_y;
extern vector<int> h_z;
extern vector<int> intensities;
extern int n;
extern int k;
extern int t;
extern string model;

void knn(vector<vector<int>>& H);
void approx_knn(vector<vector<int>>& H);
void kmeans(vector<vector<int>>& H);
void print_histograms(const vector<vector<int>>& H);

#endif
