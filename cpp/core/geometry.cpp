#include "printforge/geometry.hpp"
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
namespace printforge {
Aabb bounds(const std::vector<Vec3>& points) {
    if (points.empty()) return {};
    const auto inf = std::numeric_limits<double>::infinity();
    Aabb box{{inf, inf, inf}, {-inf, -inf, -inf}, true};
    for (const auto& p : points) {
        box.min.x = std::min(box.min.x, p.x); box.min.y = std::min(box.min.y, p.y);
        box.min.z = std::min(box.min.z, p.z); box.max.x = std::max(box.max.x, p.x);
        box.max.y = std::max(box.max.y, p.y); box.max.z = std::max(box.max.z, p.z);
    }
    return box;
}
double squared_distance(const Vec3& a, const Vec3& b) {
    const double dx=a.x-b.x, dy=a.y-b.y, dz=a.z-b.z;
    return dx*dx + dy*dy + dz*dz;
}
double point_cloud_rmse(const std::vector<Vec3>& a, const std::vector<Vec3>& b) {
    if (a.empty() || a.size()!=b.size())
        throw std::invalid_argument("point clouds must be non-empty and have equal size");
    double sum=0.0;
    for (std::size_t i=0; i<a.size(); ++i) sum += squared_distance(a[i], b[i]);
    return std::sqrt(sum/static_cast<double>(a.size()));
}
}  // namespace printforge
