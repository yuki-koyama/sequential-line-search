# Variables
set(SEQUENTIAL_LINE_SEARCH_PHOTO_NAME "hawaii.jpg" CACHE STRING "")
add_definitions(-DSEQUENTIAL_LINE_SEARCH_PHOTO_NAME=\"${SEQUENTIAL_LINE_SEARCH_PHOTO_NAME}\")
set(SEQUENTIAL_LINE_SEARCH_PHOTO_DIM_SUBSET OFF CACHE BOOL "")
if(SEQUENTIAL_LINE_SEARCH_PHOTO_DIM_SUBSET)
	add_definitions(-DSEQUENTIAL_LINE_SEARCH_PHOTO_DIM_SUBSET)
endif()

# Resources
file(GLOB shader_files ${ENHANCER_VERT_SHADER_PATH} ${ENHANCER_FRAG_SHADER_PATH})
file(GLOB photo_files ./resources/data/*.jpg)

# Qt Widgets
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTOUIC ON)
set(CMAKE_INCLUDE_CURRENT_DIR ON)

# Main source files
file(GLOB files ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp ${CMAKE_CURRENT_SOURCE_DIR}/*.h ${CMAKE_CURRENT_SOURCE_DIR}/*.ui)

# Executable definition for macOS
add_executable(sequential_line_search_photo MACOSX_BUNDLE ${files} ${shader_files} ${photo_files})
set_target_properties(sequential_line_search_photo PROPERTIES MACOSX_BUNDLE_INFO_PLIST ${CMAKE_CURRENT_SOURCE_DIR}/resources/Info.plist.in)
target_link_libraries(sequential_line_search_photo ${OPENGL_LIBRARIES} Qt5::Gui Qt5::Widgets Qt5::OpenGL Eigen3::Eigen SequentialLineSearch tinycolormap enhancer parallel-util)

# Resource properties
set_source_files_properties(
	${shader_files}
	PROPERTIES
	MACOSX_PACKAGE_LOCATION Resources/shaders
)
set_source_files_properties(
	${photo_files}
	PROPERTIES
	MACOSX_PACKAGE_LOCATION Resources/data
)
