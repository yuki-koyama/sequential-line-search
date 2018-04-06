cmake_minimum_required(VERSION 3.1)

find_package(Eigen3 REQUIRED)

include_directories(${EIGEN3_INCLUDE_DIR})

file(GLOB files *.cpp *.h)
add_library(SequentialLineSearch STATIC ${files})
target_link_libraries(SequentialLineSearch)

add_dependencies(SequentialLineSearch external_nlopt)
