#include "common.hpp"

vector<int> h_x;
vector<int> h_y;
vector<int> h_z;
vector<int> intensities;
int n;
int k;
int t;
string model;

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
