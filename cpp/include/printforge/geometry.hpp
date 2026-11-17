#pragma once
#include <vector>
namespace printforge {
struct Vec3 { double x{0.0}; double y{0.0}; double z{0.0}; };
struct Aabb { Vec3 min; Vec3 max; bool valid{false}; };
Aabb bounds(const std::vector<Vec3>& points);
double squared_distance(const Vec3& a, const Vec3& b);
double point_cloud_rmse(const std::vector<Vec3>& a, const std::vector<Vec3>& b);
}  // namespace printforge
