#include "printforge/geometry.hpp"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
namespace py = pybind11;
PYBIND11_MODULE(_printforge, module) {
    module.doc() = "Optional native geometry primitives for PrintForge3D";
    py::class_<printforge::Vec3>(module, "Vec3")
        .def(py::init<>())
        .def_readwrite("x", &printforge::Vec3::x)
        .def_readwrite("y", &printforge::Vec3::y)
        .def_readwrite("z", &printforge::Vec3::z);
    module.def("squared_distance", &printforge::squared_distance);
    module.def("point_cloud_rmse", &printforge::point_cloud_rmse);
}
