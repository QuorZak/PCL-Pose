cmake_minimum_required(VERSION 3.10 FATAL_ERROR)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

project(pose)

# PCL and Eigen are very slow without optimisation, so build in Release unless told otherwise
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# VTK (which PCL loads) refers to a JsonCpp::JsonCpp target in some distributions (e.g. VTK 9.3 on Debian 13) without
# defining it, so CMake stops with "The link interface of target VTK::jsoncpp contains JsonCpp::JsonCpp but the target
# was not found". Define it before PCL is found: from jsoncpp's own CMake package if there is one, otherwise by hand.
find_package(jsoncpp CONFIG QUIET)
if(NOT TARGET JsonCpp::JsonCpp)
  if(TARGET jsoncpp_lib)
    add_library(JsonCpp::JsonCpp INTERFACE IMPORTED)
    set_target_properties(JsonCpp::JsonCpp PROPERTIES INTERFACE_LINK_LIBRARIES jsoncpp_lib)
  else()
    find_library(JSONCPP_LIBRARY NAMES jsoncpp)
    find_path(JSONCPP_INCLUDE_DIR json/json.h PATH_SUFFIXES jsoncpp)
    if(JSONCPP_LIBRARY AND JSONCPP_INCLUDE_DIR)
      add_library(JsonCpp::JsonCpp UNKNOWN IMPORTED)
      set_target_properties(JsonCpp::JsonCpp PROPERTIES
        IMPORTED_LOCATION "${JSONCPP_LIBRARY}"
        INTERFACE_INCLUDE_DIRECTORIES "${JSONCPP_INCLUDE_DIR}")
    endif()
  endif()
endif()

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
find_package(FLANN)
include_directories(${FLANN_INCLUDE_DIRS})

include_directories(include)

# The library: everything the programs share (camera, segmentation, VFH matching, pose)
add_library(pose_core STATIC
  src/config.cpp
  src/camera.cpp
  src/segmentation.cpp
  src/vfh.cpp
  src/files.cpp
  src/library.cpp
  src/target.cpp
  src/vfh_library.cpp
  src/viewer.cpp
  src/pose.cpp)
target_include_directories(pose_core PUBLIC include)
target_link_libraries(pose_core PUBLIC ${PCL_LIBRARIES} realsense2 ${OpenCV_LIBS} ${FLANN_LIBRARIES} ${Boost_LIBRARIES})

# The programs
function(add_pose_app name source)
  add_executable(${name} ${source})
  target_link_libraries(${name} pose_core)
endfunction()

# Main executable (experimental, so it is left out of the default build: build it with `make pose`)
add_executable(pose EXCLUDE_FROM_ALL apps/main.cpp)
target_link_libraries(pose pose_core)
# Manual Measure executable
add_pose_app(manual_measure apps/manual_measure.cpp)
# Cluster Extraction executable (interactive capture, labels the object's facing angle)
add_pose_app(extraction apps/cluster_extraction.cpp)
# Cluster Extraction test executable (single capture into lab_data/test)
add_pose_app(extraction_test apps/cluster_extraction_test.cpp)
# Offline replay: runs the whole recognition pipeline on a saved scene, no camera needed
add_pose_app(replay apps/replay.cpp)
# Captures one scene from the camera and saves it for replay
add_pose_app(capture_scene apps/capture_scene.cpp)
# Library tool: checks the object library, and with --apply centres the clouds and recomputes the VFH signatures
add_pose_app(rebuild_vfh apps/rebuild_vfh.cpp)
# Scratch tools executable (a playground for trying out the functions, not an automated test suite)
add_pose_app(scratch_tools apps/scratch_tools.cpp)

# The marker programs only need the camera and OpenCV
# Marker executable (detects ArUco markers and prints their pose)
add_executable(marker apps/main_marker.cpp)
target_link_libraries(marker realsense2 ${OpenCV_LIBS})
# Marker generator executable (writes the ArUco marker images into f_markers)
add_executable(generate_markers apps/generate_markers.cpp)
target_link_libraries(generate_markers realsense2 ${OpenCV_LIBS})

# The unit tests (need GoogleTest, e.g. sudo apt install libgtest-dev). Run them with: ctest --output-on-failure
option(POSE_BUILD_TESTS "Build the unit tests" ON)
if(POSE_BUILD_TESTS)
  find_package(GTest)
  if(GTest_FOUND OR GTEST_FOUND)
    enable_testing()
    if(TARGET GTest::gtest_main)
      set(POSE_GTEST_MAIN GTest::gtest_main)
    else()
      set(POSE_GTEST_MAIN GTest::Main)
    endif()
    add_executable(pose_tests
      tests/test_camera.cpp
      tests/test_config.cpp
      tests/test_files.cpp
      tests/test_library.cpp
      tests/test_pose_manager.cpp
      tests/test_segmentation.cpp
      tests/test_target.cpp
      tests/test_vfh.cpp
      tests/test_vfh_library.cpp)
    target_link_libraries(pose_tests pose_core ${POSE_GTEST_MAIN})
    # The tests read the library and the config in the repository
    target_compile_definitions(pose_tests PRIVATE POSE_SOURCE_DIR="${CMAKE_SOURCE_DIR}")
    add_test(NAME pose_tests COMMAND pose_tests)
  else()
    message(STATUS "GoogleTest not found, so the unit tests are not built (sudo apt install libgtest-dev)")
  endif()
endif()
