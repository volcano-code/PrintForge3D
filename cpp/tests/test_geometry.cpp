#include "printforge/geometry.hpp"
#include <cassert>
#include <cmath>
#include <stdexcept>
int main() {
    using printforge::Vec3;
    const std::vector<Vec3> points{{0,1,2},{2,3,4}};
    const auto box=printforge::bounds(points);
    assert(box.valid && box.min.x==0 && box.max.x==2);
    assert(std::abs(printforge::point_cloud_rmse(points,points))<1e-12);
    bool rejected=false;
    try { (void)printforge::point_cloud_rmse(points,{}); }
    catch (const std::invalid_argument&) { rejected=true; }
    assert(rejected);
    return 0;
}
