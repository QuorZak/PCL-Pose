#pragma once
#include <string>

#include <librealsense2/rs.hpp>
#include <opencv2/objdetect/aruco_detector.hpp>

#include <Eigen/Dense>
#include <opencv2/core/eigen.hpp>
#include <opencv2/opencv.hpp>

// Parameters for fiducial markers
inline int aruco_dict_number = 2; // DICT_4X4_50=0, DICT_4X4_100=1, DICT_4X4_250=2, DICT_4X4_1000=3, DICT_5X5_50=4, DICT_5X5_100=5...
inline int aruco_marker_id = 11;
inline int aruco_marker_pixels = 500; // Only used for generating markers
inline float markerLength = 0.1f; // Marker length in meters (0.048f for the small markers, 0.1f for the large ones)
inline std::string f_markers_location = "../f_markers/";
inline std::string f_marker_name = "marker_" + std::to_string(aruco_marker_id)
      + "_dict_" + std::to_string(aruco_dict_number) + "_size_" + std::to_string(aruco_marker_pixels) + ".png";

// Global camera config params
// 848x480 resolution, 15 frames per second is optimal for Realsense D455
inline const int cam_res_width = 848; // Standard 640
inline const int cam_res_height = 480; // Standard 480
inline const int cam_fps = 15;
