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
add_executable(SequentialLineSearchPhoto MACOSX_BUNDLE ${files} ${shader_files} ${photo_files})
set_target_properties(SequentialLineSearchPhoto PROPERTIES MACOSX_BUNDLE_INFO_PLIST ${CMAKE_CURRENT_SOURCE_DIR}/resources/Info.plist.in)
target_link_libraries(SequentialLineSearchPhoto ${OPENGL_LIBRARIES} Qt5::Gui Qt5::Widgets Qt5::OpenGL Eigen3::Eigen SequentialLineSearch tinycolormap enhancer parallel-util)

# GLEW for Windows
if(WIN32)
	target_link_libraries(SequentialLineSearchPhoto ${GLEW_LIBRARY})
endif(WIN32)

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

# copy required DLLs
if(WIN32 AND MSVC)
	# Qt5 
	add_custom_command(TARGET SequentialLineSearchPhoto POST_BUILD
		COMMAND ${CMAKE_COMMAND} -E copy_if_different
			$<TARGET_FILE:Qt5::Core>
			$<TARGET_FILE:Qt5::Gui>
			$<TARGET_FILE:Qt5::Widgets>
			$<TARGET_FILE:Qt5::OpenGL>
			$<TARGET_FILE:Qt5::Concurrent>
			./$<CONFIGURATION>
	)
	# GLEW DLL (dynamic linking)
	if (NOT ${GLEW_STATIC})
		add_custom_command(TARGET SequentialLineSearchPhoto POST_BUILD
			COMMAND ${CMAKE_COMMAND} -E copy_if_different
				${GLEW_DYNAMIC_DLL}
				./$<CONFIGURATION>
		)
	endif()
endif(WIN32 AND MSVC)
