cmake_minimum_required(VERSION 3.1)

find_package(Eigen3 REQUIRED)

include_directories(${EIGEN3_INCLUDE_DIR})
include_directories(${CMAKE_SOURCE_DIR}/sequential_line_search_core)

file(GLOB files *.cpp *.h)
add_executable(SequentialLineSearch ${files})
target_link_libraries(SequentialLineSearch SequentialLineSearchCore nlopt)

add_dependencies(SequentialLineSearch external_nlopt)
