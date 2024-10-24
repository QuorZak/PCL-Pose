cmake_minimum_required(VERSION 3.10 FATAL_ERROR)

project(pose)

# PCL
find_package(PCL 1.2 REQUIRED)
include_directories(${PCL_INCLUDE_DIRS} ./)
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

# Realsense2
find_package(realsense2 REQUIRED)
include_directories(${realsense2_INCLUDE_DIRS})

# OpenCV
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# Others
find_package(HDF5)
include_directories(SYSTEM ${HDF5_INCLUDE_DIR})
find_package(FLANN)
include_directories(${FLANN_INCLUDE_DIRS})

add_executable(pose pose.cpp)

# Link libraries
target_link_libraries(pose ${realsense2_LIBRARIES} ${PCL_LIBRARIES} 
                         ${OpenCV_LIBS} ${FLANN_LIBRARIES} ${HDF5_LIBRARIES} ${Boost_LIBRARIES})

