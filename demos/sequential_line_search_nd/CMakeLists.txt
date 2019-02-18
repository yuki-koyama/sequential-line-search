include_directories(${CMAKE_SOURCE_DIR}/sequential_line_search)

file(GLOB files *.cpp *.h)
IF (WIN32)
  add_definitions(-D_USE_MATH_DEFINES)
ENDIF()
add_executable(SequentialLineSearchNd ${files})
target_link_libraries(SequentialLineSearchNd SequentialLineSearch)
