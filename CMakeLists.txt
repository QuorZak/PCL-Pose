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

#[[# ROS and MoveIt
set(catkin_DIR /opt/ros/noetic/share/catkin/cmake)
find_package(catkin REQUIRED COMPONENTS
        roscpp
        moveit_core
        moveit_ros_planning_interface
        moveit_visual_tools
)
catkin_package(INCLUDE_DIRS include
        LIBRARIES ${PROJECT_NAME}
        CATKIN_DEPENDS roscpp moveit_core moveit_ros_planning_interface moveit_visual_tools)]]
# Include catkin src directory
include_directories(src ${catkin_INCLUDE_DIRS})

# Others
find_package(HDF5)
include_directories(SYSTEM ${HDF5_INCLUDE_DIR})
find_package(FLANN)
include_directories(${FLANN_INCLUDE_DIRS})

# Main executable
add_executable(pose src/pose_estimation.h src/main.cpp)
target_link_libraries(pose ${PCL_LIBRARIES} realsense2 ${OpenCV_LIBS} ${FLANN_LIBRARIES} ${HDF5_LIBRARIES} ${Boost_LIBRARIES}) # ${catkin_LIBRARIES})

# Tests executable
add_executable(tests src/pose_estimation.h src/tests.cpp)
target_link_libraries(tests ${PCL_LIBRARIES} realsense2 ${OpenCV_LIBS} ${FLANN_LIBRARIES} ${HDF5_LIBRARIES} ${Boost_LIBRARIES}) # ${catkin_LIBRARIES})

# Cluster Extraction executable
add_executable(extraction src/pose_estimation.h src/cluster_extraction.cpp)
target_link_libraries(extraction ${PCL_LIBRARIES} realsense2 ${OpenCV_LIBS} ${FLANN_LIBRARIES} ${HDF5_LIBRARIES} ${Boost_LIBRARIES})

# Cluster Extraction executable
add_executable(extraction_test src/pose_estimation.h src/cluster_extraction_test.cpp)
target_link_libraries(extraction_test ${PCL_LIBRARIES} realsense2 ${OpenCV_LIBS} ${FLANN_LIBRARIES} ${HDF5_LIBRARIES} ${Boost_LIBRARIES})

# Main Marker executable
add_executable(marker src/pose_estimation.h src/main_marker.cpp)
target_link_libraries(marker realsense2 ${OpenCV_LIBS} ${FLANN_LIBRARIES} ${HDF5_LIBRARIES} ${Boost_LIBRARIES})

# Marker Tests executable
add_executable(marker_tests src/marker_pose_estimation.h src/marker_tests.cpp)
target_link_libraries(marker_tests realsense2 ${OpenCV_LIBS} ${FLANN_LIBRARIES} ${HDF5_LIBRARIES} ${Boost_LIBRARIES})

# Manual Measure executable
add_executable(manual_measure src/pose_estimation.h src/manual_measure.cpp)
target_link_libraries(manual_measure ${PCL_LIBRARIES} realsense2 ${OpenCV_LIBS} ${FLANN_LIBRARIES} ${HDF5_LIBRARIES} ${Boost_LIBRARIES})