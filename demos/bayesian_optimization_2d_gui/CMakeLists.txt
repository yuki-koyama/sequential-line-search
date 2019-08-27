find_package(Qt5 COMPONENTS Widgets REQUIRED)

set(CMAKE_INCLUDE_CURRENT_DIR ON)
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTOUIC ON)

file(GLOB files *.cpp *.hpp)
add_executable(BayesianOptimization2dGui ${files})
target_link_libraries(BayesianOptimization2dGui Qt5::Widgets rand-util SequentialLineSearch tinycolormap)

# set working folder as Qt5 binary folder
if(WIN32 AND MSVC)
set_property(TARGET BayesianOptimization2dGui PROPERTY VS_DEBUGGER_WORKING_DIRECTORY ${_qt5Core_install_prefix}/bin)
endif(WIN32 AND MSVC)
