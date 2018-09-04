file(GLOB files *.cpp *.h)
add_library(SequentialLineSearch STATIC ${files})
target_link_libraries(SequentialLineSearch Eigen3::Eigen)
