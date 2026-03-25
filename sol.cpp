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

void compute_cdf(vector<int>& CDF, vector<int>& Cmin) {
    for (int i = 0; i < n; i++) {
        int base = i * 256;
        if (CDF[base] > 0) Cmin[i] = CDF[base];
        for (int j = 1; j < 256; j++) {
            CDF[base + j] += CDF[base + j - 1];
            if (CDF[base + j] > 0) {
                Cmin[i] = min(Cmin[i], CDF[base + j]);
            }
        }
    }
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

void update_intensities(const vector<int>& CDF, const vector<int>& Cmin) {
    for (int i = 0; i < n; i++) {
        int base = i * 256;
        int m = (model != "kmeans") ? (k + 1) : h_cls_cnt[h_point_cls[i]];
        if (m != Cmin[i]) {
            intensities[i] = max(0, (CDF[base + intensities[i]] - Cmin[i])) * 255 / (m - Cmin[i]);
        }
    }
}

int main(int argc, char const* argv[]) {
    if (argc != 3) return 1;
    string inpfile = argv[1];
    model = argv[2];

    double st = omp_get_wtime();

    takeinput(inpfile);

    vector<int> H(1LL * n * 256, 0);

    if (model == "knn") knn(H);
    else if (model == "approx_knn") approx_knn(H);
    else if (model == "kmeans") kmeans(H);
    else return 1;

    vector<int> CDF = H;
    vector<int> Cmin(n, INT_MAX);
    compute_cdf(CDF, Cmin);

    update_intensities(CDF, Cmin);

    genopt();

    cout << "Done........." << omp_get_wtime() - st << " s" << endl;
    return 0;
}
