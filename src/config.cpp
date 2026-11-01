#include "pcl_pose/config.h"
#include "pcl_pose/parameters.h"

#include <algorithm>
#include <filesystem>
#include <iostream>

#include <opencv2/core.hpp>
#include <pcl/console/parse.h>

namespace pcl_pose {

namespace {
// Reads the key into the value if the key is in the file, otherwise leaves the value as it is
template <typename T>
void readValue(const cv::FileStorage& fs, const char* key, T& value) {
  const cv::FileNode node = fs[key];
  if (!node.empty()) {
    node >> value;
  }
}
}  // namespace

bool loadConfig(const std::string& path) {
  const cv::FileStorage fs(path, cv::FileStorage::READ);
  if (!fs.isOpened()) {
    return false;
  }

  readValue(fs, "image_reduced_to_percentage", image_reduced_to_percentage);
  readValue(fs, "depth_filter_min_distance", depth_filter_min_distance);
  readValue(fs, "depth_filter_max_distance", depth_filter_max_distance);

  readValue(fs, "cam_res_width", cam_res_width);
  readValue(fs, "cam_res_height", cam_res_height);
  readValue(fs, "cam_fps", cam_fps);

  readValue(fs, "leaf_size", leaf_size);
  readValue(fs, "cluster_tolerance", cluster_tolerance);
  readValue(fs, "min_cluster_size", min_cluster_size);
  readValue(fs, "max_cluster_size", max_cluster_size);
  readValue(fs, "segment_distance_threshold", segment_distance_threshold);
  readValue(fs, "segment_probability", segment_probability);
  readValue(fs, "segment_radius_min", segment_radius_min);
  readValue(fs, "segment_radius_max", segment_radius_max);
  readValue(fs, "filter_cluster_to_x_percent", filter_cluster_to_x_percent);

  readValue(fs, "icp_max_iterations", icp_max_iterations);
  readValue(fs, "icp_transformation_epsilon", icp_transformation_epsilon);
  readValue(fs, "icp_euclidean_fitness_epsilon", icp_euclidean_fitness_epsilon);

  readValue(fs, "start_scale_distance", start_scale_distance);
  readValue(fs, "stop_scale_distance", stop_scale_distance);
  int stability = static_cast<int>(stability_factor);
  readValue(fs, "stability_factor", stability);
  stability_factor = static_cast<PoseUpdateStabilityFactor>(std::clamp(stability, 1, 5));

  // The camera to TCP transform: only the translation is configurable (the rotation is the same as the robot frame)
  float x = camera_to_TCP(0, 3);
  float y = camera_to_TCP(1, 3);
  float z = camera_to_TCP(2, 3);
  readValue(fs, "camera_to_TCP_x", x);
  readValue(fs, "camera_to_TCP_y", y);
  readValue(fs, "camera_to_TCP_z", z);
  camera_to_TCP(0, 3) = x;
  camera_to_TCP(1, 3) = y;
  camera_to_TCP(2, 3) = z;

  readValue(fs, "spray_collision_offset", spray_collision_offset);
  readValue(fs, "object_angle_calibration_offset", object_angle_calibration_offset);
  readValue(fs, "model_directory", model_directory);

  readValue(fs, "depth_filter_smooth_alpha", depth_filter_smooth_alpha);
  readValue(fs, "depth_filter_smooth_delta", depth_filter_smooth_delta);
  readValue(fs, "depth_filter_temporal_holes_fill", depth_filter_temporal_holes_fill);
  readValue(fs, "depth_filter_holes_fill", depth_filter_holes_fill);

  return true;
}

bool loadConfigFromArguments(int argc, char** argv) {
  std::string path;
  const bool given = pcl::console::parse_argument(argc, argv, "--config", path) != -1;
  if (!given) {
    // Fall back to the default file, if there is one. Otherwise the defaults in parameters.h are used.
    path = "../config/default.yaml";
    if (!std::filesystem::exists(path)) {
      return true;
    }
  }
  if (!loadConfig(path)) {
    std::cerr << "Could not read the config file " << path << std::endl;
    return false;
  }
  std::cout << "Loaded the config from " << path << std::endl;
  return true;
}

}  // namespace pcl_pose
