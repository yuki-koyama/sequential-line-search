set(CMAKE_INCLUDE_CURRENT_DIR ON)
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTOUIC ON)

file(GLOB files
  ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/*.hpp
  ${CMAKE_CURRENT_SOURCE_DIR}/*.ui
)
add_executable(SequentialLineSearch2dGui ${files})
target_link_libraries(SequentialLineSearch2dGui Qt5::Widgets Qt5::Concurrent SequentialLineSearch tinycolormap)

# set working folder as Qt5 binary folder
if(WIN32 AND MSVC)
  set_property(TARGET SequentialLineSearch2dGui PROPERTY VS_DEBUGGER_WORKING_DIRECTORY ${_qt5Core_install_prefix}/bin)
endif(WIN32 AND MSVC)
