file(GLOB files *.cpp *.hpp)
add_executable(BayesianOptimization1d ${files})
target_link_libraries(BayesianOptimization1d SequentialLineSearch)
