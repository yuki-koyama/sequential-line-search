include_directories(${CMAKE_SOURCE_DIR}/sequential_line_search)

file(GLOB files *.cpp *.h)
add_executable(BayesianOptimization1d ${files})
target_link_libraries(BayesianOptimization1d SequentialLineSearch nlopt)
