#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
/*
// Fixes first bug on build
#include "Scene.h"
#include "Scene.cu"
// Now fails at cmake
#include "CudaSolver.h"
#include "CudaSolver.cu"
*/
#include "Environment.h"

namespace py = pybind11;

int add (int i, int j) {
    return i + j;
}

struct Pet {
    Pet(const std::string &name) : name(name) {}
    void setName(const std::string &name_) {name = name_;}
    const std::string &getName() const {return name;}
    
    std::string name;
};

struct Wrapper {
    Environment env;
    Wrapper() {env = Environment(); env.init();}
    float getState(int a) {return env.getState()[a];}
    void reset() {env.envReset();}
    void step(float r1, float r2, float r3) {env.tick(r1, r2, r3);}
};

PYBIND11_MODULE(interface, m) {
    py::class_<Wrapper>(m, "Environment")
        .def(py::init<>())
        .def("get_state", &Wrapper::getState)
        .def("reset", &Wrapper::reset)
        .def("step", &Wrapper::step);
    /*
    py::class_<Pet>(m, "Pet")
        .def(py::init<const std::string &>())
        .def("set_name", &Pet::setName)
        .def("get_name", &Pet::getName)
        .def_readwrite("name", &Pet::name)
        .def("__repr__",
            [](const Pet &a) {
                return "<example.Pet name '" + a.name + "'>";
            }
        );
    */
    /*
    m.attr("the_answer") = 42;
    py::object world = py::cast("World");
    m.attr("what") = world;
    */
    /*
    m.doc() = "pybind11 example plugin"; // optional module docstring
    m.def("add", &add, "A function which adds two numbers");
    m.def("add1", &add, py::arg("i") = 0, py::arg("j") = 0);
    */
}
