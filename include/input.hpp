#ifndef POINTCLOUD_INPUT_HPP
#define POINTCLOUD_INPUT_HPP

#include <fstream>
#include <set>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

struct InputPoint { int x, y, z, intensity; };
struct CloudInput {
    int n, k, iterations;
    std::vector<InputPoint> points;
};

inline CloudInput read_cloud(const std::string& path, const std::string& mode) {
    std::ifstream file(path);
    CloudInput cloud{};
    if (!(file >> cloud.n >> cloud.k >> cloud.iterations) || cloud.n < 1 || cloud.n > 100000 ||
        cloud.k < 1 || cloud.k > 128 || cloud.iterations < 0 || cloud.iterations > 50 ||
        cloud.k > cloud.n || (mode != "kmeans" && cloud.k >= cloud.n)) {
        throw std::runtime_error("Invalid header: require 1<=n<=100000, 1<=k<=128, 0<=T<=50; "
                                 "k<=n for kmeans, k<n for neighbors");
    }
    std::set<std::tuple<int, int, int>> coordinates;
    cloud.points.resize(cloud.n);
    for (auto& point : cloud.points) {
        if (!(file >> point.x >> point.y >> point.z >> point.intensity) ||
            point.x < -10000 || point.x > 10000 || point.y < -10000 || point.y > 10000 ||
            point.z < -10000 || point.z > 10000 || point.intensity < 0 || point.intensity > 255 ||
            !coordinates.emplace(point.x, point.y, point.z).second) {
            throw std::runtime_error("Require unique integer coordinates in [-10000,10000] "
                                     "and intensities in [0,255]");
        }
    }
    std::string extra;
    if (file >> extra) throw std::runtime_error("Unexpected data after n point records");
    return cloud;
}
#endif
