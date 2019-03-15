file(GLOB headers ${PROJECT_SOURCE_DIR}/include/sequential-line-search/*.h)
file(GLOB sources ${PROJECT_SOURCE_DIR}/src/*.cpp)
add_library(SequentialLineSearchPic STATIC ${headers} ${sources})
target_include_directories(SequentialLineSearchPic PUBLIC ${PROJECT_SOURCE_DIR}/include)
target_link_libraries(SequentialLineSearchPic Eigen3::Eigen timer nlopt nlopt-util rand-util)
target_compile_options(SequentialLineSearchPic PRIVATE -fPIC)

pybind11_add_module(pySequentialLineSearch pySequentialLineSearch.cpp)
target_link_libraries(pySequentialLineSearch PRIVATE SequentialLineSearchPic pybind11::module)
