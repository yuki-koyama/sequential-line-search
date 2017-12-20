cmake_minimum_required(VERSION 3.1)

project(SequentialLineSearch)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_BUILD_TYPE Release)

list(APPEND CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake)

add_subdirectory(sequential_line_search_core)
add_subdirectory(bayesian_optimization_1d_gui)
add_subdirectory(bayesian_optimization_2d_gui)
add_subdirectory(sequential_line_search_2d_gui)
