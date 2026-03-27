#include "common.hpp"

string model;

int main(int argc, char const* argv[]) {
    if (argc != 3) return 1;
    const string inpfile = argv[1];
    model = argv[2];

    double st = omp_get_wtime();

    if (model == "kmeans") {
        kmeans(inpfile);
    } else if (model == "knn") {
        knn(inpfile);
    } else if (model == "approx_knn") {
        approx_knn(inpfile);
    } else {
        return 1;
    }

    cout << "gpu : " << omp_get_wtime() - st << " s" << endl;
    return 0;
}
