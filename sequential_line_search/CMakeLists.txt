file(GLOB files *.cpp *.h)
add_library(SequentialLineSearch STATIC ${files})
target_include_directories(SequentialLineSearch PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(SequentialLineSearch Eigen3::Eigen timer nlopt)
