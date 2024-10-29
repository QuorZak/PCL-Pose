#pragma once
#include <librealsense2/rs.hpp>
#include <opencv2/opencv.hpp>

// Parameters for Pose Manager
inline extern float start_scale_distance = 0.01f; // Start distance for scaling in m
inline extern float stop_scale_distance = 0.5f; // Stop distance for scaling in m
enum class PoseUpdateStabilityFactor {
    VeryWilling = 1,
    Willing = 2,
    Linear = 3,
    Resistant = 4,
    VeryResistant = 5
  };
inline extern PoseUpdateStabilityFactor stability_factor = PoseUpdateStabilityFactor::Resistant;

// Parameters for fiducial markers
inline int aruco_dict_number = 2; // DICT_4X4_50=0, DICT_4X4_100=1, DICT_4X4_250=2, DICT_4X4_1000=3, DICT_5X5_50=4, DICT_5X5_100=5...
inline int aruco_marker_id = 11;
inline int aruco_marker_pixels = 500; // Only used for generating markers
inline float markerLength = 0.046f; // Marker length in meters
inline extern std::string f_markers_location = "../f_markers/";
inline extern std::string f_marker_name = "marker_" + std::to_string(aruco_marker_id)
      + "_dict_" + std::to_string(aruco_dict_number) + "_size_" + std::to_string(aruco_marker_pixels) + ".png";

// Global camera config params
// 848x480 resolution, 15 frames per second is optimal for Realsense D455
inline extern const int cam_res_width = 848; // Standard 640
inline extern const int cam_res_height = 480; // Standard 480
inline extern const int cam_fps = 15;