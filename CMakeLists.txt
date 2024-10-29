cmake_minimum_required(VERSION 3.10 FATAL_ERROR)

project(pose)

# PCL
find_package(PCL 1.2 REQUIRED)
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

# Include src directory
include_directories(src)

# Main executable
add_executable(pose src/pose_estimation.h src/main.cpp)
target_link_libraries(pose ${PCL_LIBRARIES} realsense2 ${OpenCV_LIBS} ${FLANN_LIBRARIES} ${HDF5_LIBRARIES} ${Boost_LIBRARIES})

# Tests executable
add_executable(tests src/pose_estimation.h src/tests.cpp)
target_link_libraries(tests ${PCL_LIBRARIES} realsense2 ${OpenCV_LIBS} ${FLANN_LIBRARIES} ${HDF5_LIBRARIES} ${Boost_LIBRARIES})

# Cluster Extraction executable
add_executable(extraction src/pose_estimation.h src/cluster_extraction.cpp)
target_link_libraries(extraction ${PCL_LIBRARIES} realsense2 ${OpenCV_LIBS} ${FLANN_LIBRARIES} ${HDF5_LIBRARIES} ${Boost_LIBRARIES})

# Cluster Extraction executable
add_executable(extraction_test src/pose_estimation.h src/cluster_extraction_test.cpp)
target_link_libraries(extraction_test ${PCL_LIBRARIES} realsense2 ${OpenCV_LIBS} ${FLANN_LIBRARIES} ${HDF5_LIBRARIES} ${Boost_LIBRARIES})

# Build tree executable
add_executable(build_tree src/pose_estimation.h src/build_tree.cpp)
target_link_libraries(build_tree ${PCL_LIBRARIES} realsense2 ${OpenCV_LIBS} ${FLANN_LIBRARIES} ${HDF5_LIBRARIES} ${Boost_LIBRARIES})
