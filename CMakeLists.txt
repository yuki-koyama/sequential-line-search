cmake_minimum_required(VERSION 3.1)

project(SequentialLineSearch CXX)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_BUILD_TYPE Release)

# Append possible paths to Qt5 for macOS
if(APPLE AND EXISTS /usr/local/opt/qt5)
	list(APPEND CMAKE_PREFIX_PATH "/usr/local/opt/qt5")
endif()
if(APPLE AND EXISTS /usr/local/opt/qt)
	list(APPEND CMAKE_PREFIX_PATH "/usr/local/opt/qt")
endif()

option(SEQUENTIAL_LINE_SEARCH_BUILD_COMMAND_DEMOS  "" ON )
option(SEQUENTIAL_LINE_SEARCH_BUILD_VISUAL_DEMOS   "" OFF)
option(SEQUENTIAL_LINE_SEARCH_BUILD_PHOTO_DEMOS    "" OFF)
option(SEQUENTIAL_LINE_SEARCH_BUILD_PYTHON_BINDING "" OFF)

###############################################
# Handling external libraries
###############################################

# Eigen
find_package(Eigen3 REQUIRED)
if((NOT TARGET Eigen3::Eigen) AND (DEFINED EIGEN3_INCLUDE_DIR))
	# Eigen 3.0, 3.1, 3.2 do not provide the target named Eigen3::Eigen
	add_library(AliasEigen3 INTERFACE)
	target_include_directories(AliasEigen3 INTERFACE ${EIGEN3_INCLUDE_DIR})
	add_library(Eigen3::Eigen ALIAS AliasEigen3)
endif()

# timer
set(TIMER_BUILD_TEST OFF CACHE INTERNAL "" FORCE)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/external/timer)

# nlopt
set(BUILD_SHARED_LIBS OFF CACHE INTERNAL "" FORCE)
set(NLOPT_GUILE       OFF CACHE INTERNAL "" FORCE)
set(NLOPT_LINK_PYTHON OFF CACHE INTERNAL "" FORCE)
set(NLOPT_MATLAB      OFF CACHE INTERNAL "" FORCE)
set(NLOPT_OCTAVE      OFF CACHE INTERNAL "" FORCE)
set(NLOPT_PYTHON      OFF CACHE INTERNAL "" FORCE)
set(NLOPT_SWIG        OFF CACHE INTERNAL "" FORCE)
set(NLOPT_TESTS       OFF CACHE INTERNAL "" FORCE)
set(NLOPT_CXX         ON  CACHE INTERNAL "" FORCE)
set(NLOPT_FORTRAN     OFF CACHE INTERNAL "" FORCE)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/external/nlopt)

# nlopt-util
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/external/nlopt-util)

# rand-util
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/external/rand-util)

###############################################
# Main library
###############################################

file(GLOB
	headers
	${CMAKE_CURRENT_SOURCE_DIR}/include/sequential-line-search/*.h
)
file(GLOB
	sources
	${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp
)
add_library(SequentialLineSearch STATIC ${headers} ${sources})

# for Windows
IF (WIN32)
	add_definitions(-D_USE_MATH_DEFINES)
ENDIF()

target_include_directories(SequentialLineSearch PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(SequentialLineSearch Eigen3::Eigen timer nlopt nlopt-util rand-util)

install(FILES ${headers} DESTINATION include/SequentialLineSearch)
install(TARGETS SequentialLineSearch ARCHIVE DESTINATION lib)

###############################################
# Sub-directories
###############################################

if(SEQUENTIAL_LINE_SEARCH_BUILD_COMMAND_DEMOS)
	add_subdirectory(demos/bayesian_optimization_1d)
	add_subdirectory(demos/sequential_line_search_nd)
endif()
if(SEQUENTIAL_LINE_SEARCH_BUILD_VISUAL_DEMOS)
	# Qt
	find_package(Qt5 COMPONENTS Gui Widgets Concurrent REQUIRED)

	# tinycolormap
	set(TINYCOLORMAP_WITH_EIGEN  OFF CACHE INTERNAL "" FORCE)
	set(TINYCOLORMAP_WITH_QT5    OFF CACHE INTERNAL "" FORCE)
	set(TINYCOLORMAP_BUILD_TOOLS OFF CACHE INTERNAL "" FORCE)
	add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/external/tinycolormap)

	add_subdirectory(demos/bayesian_optimization_1d_gui)
	add_subdirectory(demos/bayesian_optimization_2d_gui)
	add_subdirectory(demos/sequential_line_search_2d_gui)
endif()
if(SEQUENTIAL_LINE_SEARCH_BUILD_PHOTO_DEMOS)
	# Qt
	find_package(Qt5 COMPONENTS Gui Widgets OpenGL Concurrent REQUIRED)

	# OpenGL
	find_package(OpenGL REQUIRED)

	# tinycolormap
	if(NOT TARGET tinycolormap)
		set(TINYCOLORMAP_BUILD_TOOLS OFF CACHE INTERNAL "" FORCE)
		add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/external/tinycolormap)
	endif()

	# enhancer
	set(ENHANCER_USE_QT_FEATURES ON CACHE INTERNAL "" FORCE)
	set(ENHANCER_BUILD_QT_TESTS OFF CACHE INTERNAL "" FORCE)
	add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/external/enhancer)

	# parallel-util
	set(PARALLELUTIL_BUILD_TESTS OFF CACHE INTERNAL "" FORCE)
	add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/external/parallel-util)

	add_subdirectory(demos/sequential_line_search_photo)
endif()
if(SEQUENTIAL_LINE_SEARCH_BUILD_PYTHON_BINDING)
	# pybind11
	set(PYBIND11_INSTALL OFF CACHE INTERNAL "" FORCE)
	set(PYBIND11_TEST    OFF CACHE INTERNAL "" FORCE)
	add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/external/pybind11)

	add_subdirectory(python)
endif()

###############################################
# Tests
###############################################
enable_testing()
if(SEQUENTIAL_LINE_SEARCH_BUILD_COMMAND_DEMOS)
	add_test(NAME bayesian_optimization_1d_test COMMAND $<TARGET_FILE:BayesianOptimization1d>)
	add_test(NAME sequential_line_search_nd_test COMMAND $<TARGET_FILE:SequentialLineSearchNd>)
endif()
