file(GLOB files *.cpp *.h)
add_executable(SequentialLineSearchNd ${files})
target_link_libraries(SequentialLineSearchNd SequentialLineSearch)
