// python 3.7 deprecate PyCreateThread, but pybind11 2.2.3 still uses
// this function.

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "DPU.hpp"
//#include "ResNet50.hpp"

namespace py = pybind11;

PYBIND11_MODULE(dpu, m) 
{
    // Description
    m.doc() = "Python wrapper for Inference";
    // Class DPU
    py::class_<DPU>(m, "DPU")
        // Constructor
        .def(py::init<>())
        // Class methods
        .def("Load", &DPU::Load, py::arg("model_filename"), "Load model")
        .def("Run", &DPU::Run, py::arg("image_filepath"), "Run infer");
}

/*
PYBYND11_MODULE(resnet50, m)
{
    // Class ResNet50
    py::class<ResNet50>(m, "ResNet50")
        .def(py::init<>());
}
*/
