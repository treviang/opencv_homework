cmake_minimum_required(VERSION 2.8)

project(StereoReconstruction)

find_package(OpenCV REQUIRED)
find_package(PCL REQUIRED)

message(STATUS "OpenCV include dirs found: ${OpenCV_INCLUDE_DIRS}")

include_directories(
	include
	${PCL_INCLUDE_DIRS}
	${OpenCV_INCLUDE_DIRS}
)

add_definitions(${PCL_DEFINITIONS})

SET(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)

add_executable(calibrate 
	src/calibrate.cpp 
	include/calibrate.h) 
# header files are not required but add them in order get them shown in
# visual studio

target_link_libraries(calibrate ${OpenCV_LIBRARIES})
	
add_executable(reconstruct 
	src/reconstruct.cpp
	include/reconstruct.h)

target_link_libraries(reconstruct ${OpenCV_LIBRARIES} ${PCL_LIBRARIES})
