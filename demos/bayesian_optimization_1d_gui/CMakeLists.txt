cmake_minimum_required(VERSION 3.1)

find_package(Eigen3 REQUIRED)
find_package(Qt5 COMPONENTS Widgets REQUIRED)

set(CMAKE_INCLUDE_CURRENT_DIR ON)
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTOUIC ON)

include_directories(${EIGEN3_INCLUDE_DIR})
include_directories(${CMAKE_SOURCE_DIR}/sequential_line_search)

file(GLOB files *.cpp *.h)
add_executable(BayesianOptimization1dGui ${files})
target_link_libraries(BayesianOptimization1dGui Qt5::Widgets SequentialLineSearch nlopt)

add_dependencies(BayesianOptimization1dGui external_nlopt)
