#pragma once
#include <string>

#include <Eigen/Dense>

namespace pcl_pose {

// Declare global variables that all functions in the library and the programs can access
inline const int image_reduced_to_percentage = 80;
inline const float depth_filter_min_distance = 0.25f;
inline const float depth_filter_max_distance = 0.7f;

// Global camera config params
// 848x480 resolution, 15 frames per second is optimal for Realsense D455
inline const int cam_res_width = 848; // Standard 640
inline const int cam_res_height = 480; // Standard 480
inline const int cam_fps = 15;

// Parameters for cloud filtering
inline const float leaf_size = 0.004f; // 0.01f default
inline const float cluster_tolerance = 0.010f; // 0.02 default
inline const int min_cluster_size = 500; // 100 default
inline const int max_cluster_size = 25000; // 25000 default
inline const float segment_distance_threshold = 0.02f; // 0.02 default
inline const float segment_probability = 0.99f; // try 0.99 ? Increase the probability to get a good sample
inline const float segment_radius_min = 0.01f; // try 0.01 ? Set radius limits to avoid collinear points
inline const float segment_radius_max = 0.1f; // try 0.1 ?
inline const float filter_cluster_to_x_percent = 0.99f; // 0.4 default

// Parameters for Pose Estimation
inline int icp_max_iterations = 50; // Maximum number of ICP iterations
inline float icp_transformation_epsilon = 1e-8; // Transformation epsilon for ICP convergence
inline float icp_euclidean_fitness_epsilon = 1; // Euclidean fitness epsilon for ICP convergence

// Parameters for Pose Manager
inline float start_scale_distance = 0.01f; // Start distance for scaling in m
inline float stop_scale_distance = 0.5f; // Stop distance for scaling in m
enum class PoseUpdateStabilityFactor {
  VeryWilling = 1,
  Willing = 2,
  Linear = 3,
  Resistant = 4,
  VeryResistant = 5
};
inline PoseUpdateStabilityFactor stability_factor = PoseUpdateStabilityFactor::Resistant;

// Define the transform to convert from camera frame to robot end effector frame (TCP)
// For now was manually set. The front, middle surface of the camera is 0 x, -60mm y, +31mm z, from the robot frame
// The calibration of the camera means that it is aligned to the left lens. To get the center shift 47.5mm (cam baseline is 95mm)
// The rotation is the same as the robot frame
inline const Eigen::Matrix4f camera_to_TCP = (Eigen::Matrix4f() <<
  1, 0, 0, -0.0475,
  0, 1, 0, -0.06,
  0, 0, 1, 0.031,
  0, 0, 0, 1).finished();
// Some documentation alludes to the ground truth adjustment for the depth reading regarding Intel D455 being -4.55mm z from the front surface

// The offset for the end effector attachment and it's collision with different objects
inline float spray_collision_offset = -57.0f;

//create an enum called home_position
enum class HomePosition {
  ELEVATED = 1,
  FRONT = 2,
  ANY = 3
};
////////////////////////////   TCP ELEVATED (home reference)   /////////////////////////////////////////////////////////
// The position of the TCP ELEVATED (home reference) from the base plate (world reference) is:
// x = 305.78, y = 338.90, z = 73.94 mm
// Rotational Vector (Rodrigues), Rotation x = 3.613 rad, Rotation y = -1.863 rad, Rotation z = 0.986 rad
// Camera rotation for TCP ELEVATED (home reference): the rotation (RX) that needs to be applied around the x-axis of the
// tool (TCP) frame each time the camera is moved from here. It is calculated at run time from the camera's accelerometer
// (see align_yaxis_with_gravity).
// Object calibration destination from TCP ELEVATED is:
// x = 0, y = 20, z = 360 mm
// RX = the rotation calculated from the accelerometer,
/////////////////////////////////   TCP FRONT (home reference)   ///////////////////////////////////////////////////////
// The same values for the TCP FRONT (home reference) point from the base plate (world reference) are:
// x = 260, y = 325, z = -90 mm
// Rotational Vector (Rodrigues), Rotation x = 216.67 rad, Rotation y = -111.97 rad, Rotation z = 0.94.82 rad
// Object calibration destination from TCP FRONT is:
// x = 0, y = 10, z = 380 mm

// Directory for reference models
inline std::string model_directory = "../lab_data/spray_bottle_tall/"; // "../lab_data/";

// Values for the depth post-processing filters (see camera.h)
inline float depth_filter_smooth_alpha = 0.25f; // Smoothing factor 0.25
inline float depth_filter_smooth_delta = 60; // Delta value 60
inline float depth_filter_temporal_holes_fill = 6.0f; // Persistency index (highest = 8) [0-8]
inline float depth_filter_holes_fill = 2.0f;// 2 = fill from the farthest pixel

}  // namespace pcl_pose
