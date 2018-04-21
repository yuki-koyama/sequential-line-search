cmake_minimum_required(VERSION 3.1)

project(SequentialLineSearch CXX)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_BUILD_TYPE Release)

###############################################
# Handling external libraries
###############################################
find_package(Qt5 COMPONENTS Widgets)
if(Qt5_FOUND)
	include_directories(${Qt5Widgets_INCLUDE_DIRS})
endif()
find_package(Eigen3 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIR})

set(ExternalInstallDir "${CMAKE_BINARY_DIR}/external")
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
# Building external libraries
###############################################
include(ExternalProject)
ExternalProject_Add(
	nlopt
	INSTALL_DIR ${ExternalInstallDir}/
	GIT_REPOSITORY https://github.com/stevengj/nlopt.git
	GIT_TAG eafdc8ce9403a4e1532239c1bebbdb86064eda38
	CMAKE_ARGS 
	-DCMAKE_INSTALL_PREFIX:PATH=<INSTALL_DIR>
	-DBUILD_SHARED_LIBS:BOOL=0
	-DNLOPT_GUILE:BOOL=0
	-DNLOPT_SWIG:BOOL=0
	-DNLOPT_PYTHON:BOOL=0
	-DNLOPT_MATLAB:BOOL=0
	-DNLOPT_OCTAVE:BOOL=0
)
ExternalProject_Add(
	tinycolormap
	PREFIX ${ExternalInstallDir}/
	INSTALL_DIR ${ExternalInstallDir}/
	GIT_REPOSITORY https://github.com/yuki-koyama/tinycolormap.git
	GIT_TAG v0.1.0
	CMAKE_ARGS -DCMAKE_INSTALL_PREFIX:PATH=<INSTALL_DIR>
)
add_dependencies(SequentialLineSearch nlopt tinycolormap)

###############################################
# Tests
###############################################
enable_testing()
add_test(NAME bayesian_optimization_1d_test COMMAND $<TARGET_FILE:BayesianOptimization1d>)
add_test(NAME sequential_line_search_nd_test COMMAND $<TARGET_FILE:SequentialLineSearchNd>)
