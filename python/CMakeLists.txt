IF (WIN32)
  add_definitions(-D_USE_MATH_DEFINES)
ENDIF()

pybind11_add_module(pySequentialLineSearch pySequentialLineSearch.cpp)
target_link_libraries(pySequentialLineSearch PRIVATE SequentialLineSearch Eigen3::Eigen timer nlopt nlopt-util rand-util pybind11::module)
