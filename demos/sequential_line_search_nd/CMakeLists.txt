file(GLOB files *.cpp *.hpp)
add_executable(SequentialLineSearchNd ${files})
target_link_libraries(SequentialLineSearchNd SequentialLineSearch)
