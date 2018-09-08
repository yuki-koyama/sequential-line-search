include_directories(${CMAKE_SOURCE_DIR}/sequential_line_search)

file(GLOB files *.cpp *.h)
add_executable(SequentialLineSearchNd ${files})
target_link_libraries(SequentialLineSearchNd SequentialLineSearch)
