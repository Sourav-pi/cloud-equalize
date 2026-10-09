#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <limits>
#include <numeric>
#include <string>
#include <tuple>
#include <vector>
#include "input.hpp"

using namespace std;

struct Point {
    long long x;
    long long y;
    long long z;
    int intensity;
};

static int n = 0;
static int k = 0;
static int t = 0;

static vector<Point> points;

static inline long long sq_dist(const Point& a, const Point& b) {
    long long dx = a.x - b.x;
    long long dy = a.y - b.y;
    long long dz = a.z - b.z;
    return dx * dx + dy * dy + dz * dz;
}

static bool read_input(const string& path) {
    try {
        const auto cloud = read_cloud(path, "kmeans");
        n = cloud.n; k = cloud.k; t = cloud.iterations;
        points.resize(n);
        for (int i = 0; i < n; ++i) {
            const auto& p = cloud.points[i];
            points[i] = {p.x, p.y, p.z, p.intensity};
        }
    } catch (const exception& ex) {
        cerr << ex.what() << '\n';
        return false;
    }
    return true;
}

static inline int remap_intensity(int old_i, const vector<int>& cdf, int cmin, int m) {
    if (m == cmin) return old_i;
    int num = cdf[old_i] - cmin;
    if (num < 0) num = 0;
    return (num * 255) / (m - cmin);
}

static inline pair<vector<int>, int> cdf_and_cmin(const vector<int>& hist) {
    vector<int> cdf(256, 0);
    int cmin = numeric_limits<int>::max();
    int run = 0;
    for (int v = 0; v < 256; ++v) {
        run += hist[v];
        cdf[v] = run;
        if (cdf[v] > 0) cmin = min(cmin, cdf[v]);
    }
    if (cmin == numeric_limits<int>::max()) cmin = 0;
    return {cdf, cmin};
}

static vector<int> knn_equalized() {
    if (k > n - 1) {
        throw runtime_error("KNN requires k <= n-1");
    }

    vector<int> out(n, 0);
    vector<pair<long long, int>> d;
    d.reserve(max(0, n - 1));

    for (int i = 0; i < n; ++i) {
        d.clear();
        for (int j = 0; j < n; ++j) {
            if (j == i) continue;
            d.push_back({sq_dist(points[i], points[j]), j});
        }
        sort(d.begin(), d.end(), [](const pair<long long, int>& a, const pair<long long, int>& b) {
            if (a.first != b.first) return a.first < b.first;
            return std::tie(points[a.second].x, points[a.second].y, points[a.second].z, a.second)
                 < std::tie(points[b.second].x, points[b.second].y, points[b.second].z, b.second);
        });

        vector<int> hist(256, 0);
        hist[points[i].intensity]++;
        for (int r = 0; r < k; ++r) {
            ++hist[points[d[r].second].intensity];
        }

        vector<int> cdf;
        int cmin;
        std::tie(cdf, cmin) = cdf_and_cmin(hist);
        out[i] = remap_intensity(points[i].intensity, cdf, cmin, k + 1);
    }

    return out;
}

static vector<int> approx_knn_equalized() {
    // Deterministic fallback approximation for baseline benchmarking.
    // Uses exact KNN to keep baseline correctness simple and stable.
    return knn_equalized();
}

static vector<int> kmeans_assignments() {
    if (k > n) {
        throw runtime_error("K-means requires k <= n");
    }

    vector<array<long long, 3>> centroids(k);
    for (int i = 0; i < k; ++i) {
        centroids[i] = {points[i].x, points[i].y, points[i].z};
    }

    vector<int> assign(n, -1);
    for (int iter = 0; iter < t; ++iter) {
        bool changed = false;
        vector<int> next_assign(n, 0);

        for (int i = 0; i < n; ++i) {
            long long best_d = numeric_limits<long long>::max();
            int best_c = 0;
            for (int c = 0; c < k; ++c) {
                long long dx = points[i].x - centroids[c][0];
                long long dy = points[i].y - centroids[c][1];
                long long dz = points[i].z - centroids[c][2];
                long long dsq = dx * dx + dy * dy + dz * dz;
                
                if (dsq < best_d) {
                    best_d = dsq;
                    best_c = c;
                } else if (dsq == best_d) {
                    // Lexicographical tie-breaker (X, then Y, then Z)
                    long long cx = centroids[c][0];
                    long long cy = centroids[c][1];
                    long long cz = centroids[c][2];
                    
                    long long best_cx = centroids[best_c][0];
                    long long best_cy = centroids[best_c][1];
                    long long best_cz = centroids[best_c][2];

                    if (cx < best_cx || 
                       (cx == best_cx && cy < best_cy) || 
                       (cx == best_cx && cy == best_cy && cz < best_cz)) {
                        best_c = c;
                    }
                }
            }
            next_assign[i] = best_c;
            if (assign[i] != best_c) changed = true;
        }

        assign.swap(next_assign);
        if (!changed) break;

        vector<long long> sx(k, 0), sy(k, 0), sz(k, 0);
        vector<int> cnt(k, 0);
        for (int i = 0; i < n; ++i) {
            int c = assign[i];
            sx[c] += points[i].x;
            sy[c] += points[i].y;
            sz[c] += points[i].z;
            cnt[c] += 1;
        }

        for (int c = 0; c < k; ++c) {
            if (cnt[c] == 0) continue;
            centroids[c][0] = sx[c] / cnt[c];
            centroids[c][1] = sy[c] / cnt[c];
            centroids[c][2] = sz[c] / cnt[c];
        }
    }

    if (assign[0] == -1) {
        // t can be 0 in general input variants.
        for (int i = 0; i < n; ++i) {
            long long best_d = numeric_limits<long long>::max();
            int best_c = 0;
            for (int c = 0; c < k; ++c) {
                long long dx = points[i].x - centroids[c][0];
                long long dy = points[i].y - centroids[c][1];
                long long dz = points[i].z - centroids[c][2];
                long long dsq = dx * dx + dy * dy + dz * dz;
                
                if (dsq < best_d) {
                    best_d = dsq;
                    best_c = c;
                } else if (dsq == best_d) {
                    // Lexicographical tie-breaker (X, then Y, then Z)
                    long long cx = centroids[c][0];
                    long long cy = centroids[c][1];
                    long long cz = centroids[c][2];
                    
                    long long best_cx = centroids[best_c][0];
                    long long best_cy = centroids[best_c][1];
                    long long best_cz = centroids[best_c][2];

                    if (cx < best_cx || 
                       (cx == best_cx && cy < best_cy) || 
                       (cx == best_cx && cy == best_cy && cz < best_cz)) {
                        best_c = c;
                    }
                }
            }
            assign[i] = best_c;
        }
    }

    return assign;
}

static vector<int> kmeans_equalized() {
    vector<int> assign = kmeans_assignments();

    vector<vector<int>> cluster_hist(k, vector<int>(256, 0));
    vector<int> cluster_size(k, 0);

    for (int i = 0; i < n; ++i) {
        int c = assign[i];
        ++cluster_hist[c][points[i].intensity];
        ++cluster_size[c];
    }

    vector<int> out(n, 0);
    for (int i = 0; i < n; ++i) {
        int c = assign[i];
        vector<int> cdf;
        int cmin;
        std::tie(cdf, cmin) = cdf_and_cmin(cluster_hist[c]);
        out[i] = remap_intensity(points[i].intensity, cdf, cmin, cluster_size[c]);
    }

    return out;
}

static bool write_output(const string& path, const vector<int>& new_i) {
    ofstream out(path);
    if (!out) return false;

    for (int i = 0; i < n; ++i) {
        out << points[i].x << ' ' << points[i].y << ' ' << points[i].z << ' ' << new_i[i] << '\n';
    }
    out.close();
    return bool(out);
}

int main(int argc, char** argv) {
    if (argc < 3 || argc > 4) {
        cerr << "Usage: " << argv[0] << " <input_file> <knn|approx_knn|kmeans> [output_file]\n";
        return 1;
    }
    const auto start_time = chrono::steady_clock::now();

    const string input_file = argv[1];
    const string model = argv[2];
    const string output_file = (argc == 4) ? argv[3] : ("cpu_" + model + ".txt");

    if (!read_input(input_file)) {
        cerr << "Failed to parse input file: " << input_file << '\n';
        return 1;
    }

    try {
        vector<int> out;
        if (model == "knn") {
            out = knn_equalized();
        } else if (model == "approx_knn") {
            out = approx_knn_equalized();
        } else if (model == "kmeans") {
            out = kmeans_equalized();
        } else {
            cerr << "Unknown model: " << model << '\n';
            return 1;
        }
        
        if (!write_output(output_file, out)) {
            cerr << "Failed to write output file: " << output_file << '\n';
            return 1;
        }
    } catch (const exception& ex) {
        cerr << "Error: " << ex.what() << '\n';
        return 1;
    }
    const double elapsed = chrono::duration<double>(chrono::steady_clock::now() - start_time).count();
    std::cout << "CPU Walltime: " << elapsed << " seconds\n";

    return 0;
}
