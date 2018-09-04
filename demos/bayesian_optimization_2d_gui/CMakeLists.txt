find_package(Qt5 COMPONENTS Widgets REQUIRED)

set(CMAKE_INCLUDE_CURRENT_DIR ON)
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTOUIC ON)

include_directories(${CMAKE_SOURCE_DIR}/sequential_line_search)

file(GLOB files *.cpp *.h)
add_executable(BayesianOptimization2dGui ${files})
target_link_libraries(BayesianOptimization2dGui Qt5::Widgets SequentialLineSearch nlopt tinycolormap)
