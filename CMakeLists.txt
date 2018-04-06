cmake_minimum_required(VERSION 3.1)

project(SequentialLineSearch CXX)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_BUILD_TYPE Release)

find_package(Qt5 COMPONENTS Widgets)

###############################################
# External libraries
###############################################
set(ExternalInstallDir "${CMAKE_BINARY_DIR}/external")
include(ExternalProject)
ExternalProject_Add(
	external_nlopt
	INSTALL_DIR ${ExternalInstallDir}/
	GIT_REPOSITORY https://github.com/yuki-koyama/nlopt.git
	GIT_TAG master
	CMAKE_ARGS -DCMAKE_INSTALL_PREFIX:PATH=<INSTALL_DIR> -DBUILD_SHARED_LIBS:BOOL=0 -DNLOPT_GUILE:BOOL=0 -DNLOPT_SWIG:BOOL=0 -DNLOPT_PYTHON:BOOL=0 -DNLOPT_MATLAB:BOOL=0 -DNLOPT_OCTAVE:BOOL=0
)
ExternalProject_Add(
	external_tinycolormap
	PREFIX ${ExternalInstallDir}/
	INSTALL_DIR ${ExternalInstallDir}/
	GIT_REPOSITORY https://github.com/yuki-koyama/tinycolormap.git
	GIT_TAG master
	CMAKE_ARGS -DCMAKE_INSTALL_PREFIX:PATH=<INSTALL_DIR>
)
link_directories(${ExternalInstallDir}/lib)
include_directories(${ExternalInstallDir}/include)

###############################################
# Sub-directories
###############################################
add_subdirectory(sequential_line_search)
add_subdirectory(demos/bayesian_optimization_1d)
add_subdirectory(demos/sequential_line_search_nd)
if(Qt5_FOUND)
	add_subdirectory(demos/bayesian_optimization_1d_gui)
	add_subdirectory(demos/bayesian_optimization_2d_gui)
	add_subdirectory(demos/sequential_line_search_2d_gui)
endif()

###############################################
# Tests
###############################################
enable_testing()
add_test(NAME bayesian_optimization_1d_test COMMAND $<TARGET_FILE:BayesianOptimization1d>)
add_test(NAME sequential_line_search_nd_test COMMAND $<TARGET_FILE:SequentialLineSearchNd>)

