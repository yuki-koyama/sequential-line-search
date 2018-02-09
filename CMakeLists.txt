cmake_minimum_required(VERSION 3.1)
enable_testing()

project(SequentialLineSearch)

find_package(Qt5 COMPONENTS Widgets)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_BUILD_TYPE Release)

list(APPEND CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake)

add_subdirectory(sequential_line_search_core)
add_subdirectory(bayesian_optimization_1d)
add_subdirectory(sequential_line_search)
if(Qt5_FOUND)
	add_subdirectory(bayesian_optimization_1d_gui)
	add_subdirectory(bayesian_optimization_2d_gui)
	add_subdirectory(sequential_line_search_2d_gui)
endif()

add_test(NAME bayesian_optimization_1d_test COMMAND $<TARGET_FILE:BayesianOptimization1d>)
add_test(NAME sequential_line_search_test COMMAND $<TARGET_FILE:SequentialLineSearch>)
