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

include_directories(include)

# The library: everything the programs share (camera, segmentation, VFH matching, pose)
add_library(pose_core STATIC
  src/camera.cpp
  src/segmentation.cpp
  src/vfh.cpp
  src/files.cpp
  src/viewer.cpp
  src/pose.cpp)
target_include_directories(pose_core PUBLIC include)
target_link_libraries(pose_core PUBLIC ${PCL_LIBRARIES} realsense2 ${OpenCV_LIBS} ${FLANN_LIBRARIES} ${HDF5_LIBRARIES} ${Boost_LIBRARIES})

# The programs
function(add_pose_app name source)
  add_executable(${name} ${source})
  target_link_libraries(${name} pose_core)
endfunction()

# Main executable (experimental, does not build yet, so it is left out of the default build)
add_executable(pose EXCLUDE_FROM_ALL apps/main.cpp)
target_link_libraries(pose pose_core)
# Manual Measure executable
add_pose_app(manual_measure apps/manual_measure.cpp)
# Cluster Extraction executable (interactive capture, labels the object's facing angle)
add_pose_app(extraction apps/cluster_extraction.cpp)
# Cluster Extraction test executable (single capture into lab_data/test)
add_pose_app(extraction_test apps/cluster_extraction_test.cpp)
# Scratch tools executable (a playground for trying out the functions, not an automated test suite)
add_pose_app(scratch_tools apps/scratch_tools.cpp)

# The marker programs only need the camera and OpenCV
# Marker executable (detects ArUco markers and prints their pose)
add_executable(marker apps/main_marker.cpp)
target_link_libraries(marker realsense2 ${OpenCV_LIBS})
# Marker generator executable (writes the ArUco marker images into f_markers)
add_executable(generate_markers apps/generate_markers.cpp)
target_link_libraries(generate_markers realsense2 ${OpenCV_LIBS})
