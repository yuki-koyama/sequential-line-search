cmake_minimum_required(VERSION 3.1)

find_package(Eigen3 REQUIRED)
find_package(NLopt REQUIRED)

include_directories(${EIGEN3_INCLUDE_DIR})
include_directories(${NLopt_INCLUDE_DIRS})
include_directories(${CMAKE_SOURCE_DIR}/sequential_line_search_core)

file(GLOB files *.cpp *.h)
add_executable(BayesianOptimization1d ${files})
target_link_libraries(BayesianOptimization1d ${NLopt_LIBRARIES} SequentialLineSearchCore)