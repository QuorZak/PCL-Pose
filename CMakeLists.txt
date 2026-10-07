cmake_minimum_required(VERSION 3.10 FATAL_ERROR)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

project(pose)

# PCL
find_package(PCL 1.8 REQUIRED)
include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

# Realsense2
find_package(realsense2 REQUIRED)
include_directories(${realsense2_INCLUDE_DIRS})

# OpenCV
find_package(OpenCV 4.7 REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# Others
find_package(HDF5)
include_directories(SYSTEM ${HDF5_INCLUDE_DIR})
find_package(FLANN)
include_directories(${FLANN_INCLUDE_DIRS})

include_directories(src)

# Main executable
add_executable(pose src/pose_estimation.h src/main.cpp)
target_link_libraries(pose ${PCL_LIBRARIES} realsense2 ${OpenCV_LIBS} ${FLANN_LIBRARIES} ${HDF5_LIBRARIES} ${Boost_LIBRARIES})

# Scratch tools executable (a playground for trying out the functions, not an automated test suite)
add_executable(scratch_tools src/pose_estimation.h src/scratch_tools.cpp)
target_link_libraries(scratch_tools ${PCL_LIBRARIES} realsense2 ${OpenCV_LIBS} ${FLANN_LIBRARIES} ${HDF5_LIBRARIES} ${Boost_LIBRARIES})

# Cluster Extraction executable (interactive capture, labels the object's facing angle)
add_executable(extraction src/pose_estimation.h src/cluster_extraction.cpp)
target_link_libraries(extraction ${PCL_LIBRARIES} realsense2 ${OpenCV_LIBS} ${FLANN_LIBRARIES} ${HDF5_LIBRARIES} ${Boost_LIBRARIES})

# Cluster Extraction test executable (single capture into lab_data/test)
add_executable(extraction_test src/pose_estimation.h src/cluster_extraction_test.cpp)
target_link_libraries(extraction_test ${PCL_LIBRARIES} realsense2 ${OpenCV_LIBS} ${FLANN_LIBRARIES} ${HDF5_LIBRARIES} ${Boost_LIBRARIES})

# Marker executable (detects ArUco markers and prints their pose)
add_executable(marker src/marker_pose_estimation.h src/main_marker.cpp)
target_link_libraries(marker realsense2 ${OpenCV_LIBS} ${Boost_LIBRARIES})

# Marker generator executable (writes the ArUco marker images into f_markers)
add_executable(generate_markers src/marker_pose_estimation.h src/generate_markers.cpp)
target_link_libraries(generate_markers realsense2 ${OpenCV_LIBS} ${Boost_LIBRARIES})

# Manual Measure executable
add_executable(manual_measure src/pose_estimation.h src/manual_measure.cpp)
target_link_libraries(manual_measure ${PCL_LIBRARIES} realsense2 ${OpenCV_LIBS} ${FLANN_LIBRARIES} ${HDF5_LIBRARIES} ${Boost_LIBRARIES})
