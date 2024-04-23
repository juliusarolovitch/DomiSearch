#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "astar.cpp"  // Include your A* implementation file

namespace py = pybind11;

PYBIND11_MODULE(astar_module, m) {
    m.doc() = "A* module implemented in C++";
    m.def("astar", &astar, "A function that performs A* search");
}
