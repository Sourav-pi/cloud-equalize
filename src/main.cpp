#include "common.hpp"

string output_path;

int main(int argc, char const* argv[]) {
    if (argc < 3 || argc > 4) {
        cerr << "Usage: " << argv[0] << " <input> <knn|approx_knn|kmeans> [output]\n";
        return 1;
    }
    const string inpfile = argv[1];
    const string model = argv[2];
    output_path = argc == 4 ? argv[3] : model + ".txt";

    double st = omp_get_wtime();

    try {
        if (model == "kmeans") {
            kmeans(inpfile);
        } else if (model == "knn") {
            knn(inpfile);
        } else if (model == "approx_knn") {
            approx_knn(inpfile);
        } else {
            throw runtime_error("Unknown model: " + model);
        }
    } catch (const exception& ex) {
        cerr << "Error: " << ex.what() << '\n';
        return 1;
    }

    cout << "gpu : " << omp_get_wtime() - st << " s" << endl;
    return 0;
}
