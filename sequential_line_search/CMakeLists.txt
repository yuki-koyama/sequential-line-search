file(GLOB
	headers
	${CMAKE_CURRENT_SOURCE_DIR}/*.h
)
file(GLOB
	sources
	${CMAKE_CURRENT_SOURCE_DIR}/*.cpp
)
add_library(SequentialLineSearch STATIC ${headers} ${sources})
target_include_directories(SequentialLineSearch PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(SequentialLineSearch Eigen3::Eigen timer nlopt)

install(FILES ${headers} DESTINATION include/SequentialLineSearch)
install(TARGETS SequentialLineSearch ARCHIVE DESTINATION lib)
