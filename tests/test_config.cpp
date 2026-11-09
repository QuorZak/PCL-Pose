#include <gtest/gtest.h>

#include <filesystem>
#include <fstream>
#include <string>

#include "pcl_pose/config.h"
#include "pcl_pose/parameters.h"

using namespace pcl_pose;
namespace fs = std::filesystem;

namespace {

// All the values that the config file can set
struct Parameters {
  int image_reduced_to_percentage, cam_res_width, cam_res_height, cam_fps, min_cluster_size, max_cluster_size,
      icp_max_iterations;
  float depth_filter_min_distance, depth_filter_max_distance, leaf_size, cluster_tolerance,
      segment_distance_threshold, segment_probability, segment_radius_min, segment_radius_max,
      filter_cluster_to_x_percent, icp_transformation_epsilon, icp_euclidean_fitness_epsilon, start_scale_distance,
      stop_scale_distance, spray_collision_offset, object_angle_calibration_offset, depth_filter_smooth_alpha,
      depth_filter_smooth_delta, depth_filter_temporal_holes_fill, depth_filter_holes_fill;
  PoseUpdateStabilityFactor stability_factor;
  Eigen::Matrix4f camera_to_TCP;
  std::string model_directory;
};

Parameters currentParameters() {
  return {image_reduced_to_percentage, cam_res_width, cam_res_height, cam_fps, min_cluster_size, max_cluster_size,
          icp_max_iterations, depth_filter_min_distance, depth_filter_max_distance, leaf_size, cluster_tolerance,
          segment_distance_threshold, segment_probability, segment_radius_min, segment_radius_max,
          filter_cluster_to_x_percent, icp_transformation_epsilon, icp_euclidean_fitness_epsilon,
          start_scale_distance, stop_scale_distance, spray_collision_offset, object_angle_calibration_offset,
          depth_filter_smooth_alpha, depth_filter_smooth_delta, depth_filter_temporal_holes_fill,
          depth_filter_holes_fill, stability_factor, camera_to_TCP, model_directory};
}

void setParameters(const Parameters& p) {
  image_reduced_to_percentage = p.image_reduced_to_percentage;
  cam_res_width = p.cam_res_width;
  cam_res_height = p.cam_res_height;
  cam_fps = p.cam_fps;
  min_cluster_size = p.min_cluster_size;
  max_cluster_size = p.max_cluster_size;
  icp_max_iterations = p.icp_max_iterations;
  depth_filter_min_distance = p.depth_filter_min_distance;
  depth_filter_max_distance = p.depth_filter_max_distance;
  leaf_size = p.leaf_size;
  cluster_tolerance = p.cluster_tolerance;
  segment_distance_threshold = p.segment_distance_threshold;
  segment_probability = p.segment_probability;
  segment_radius_min = p.segment_radius_min;
  segment_radius_max = p.segment_radius_max;
  filter_cluster_to_x_percent = p.filter_cluster_to_x_percent;
  icp_transformation_epsilon = p.icp_transformation_epsilon;
  icp_euclidean_fitness_epsilon = p.icp_euclidean_fitness_epsilon;
  start_scale_distance = p.start_scale_distance;
  stop_scale_distance = p.stop_scale_distance;
  spray_collision_offset = p.spray_collision_offset;
  object_angle_calibration_offset = p.object_angle_calibration_offset;
  depth_filter_smooth_alpha = p.depth_filter_smooth_alpha;
  depth_filter_smooth_delta = p.depth_filter_smooth_delta;
  depth_filter_temporal_holes_fill = p.depth_filter_temporal_holes_fill;
  depth_filter_holes_fill = p.depth_filter_holes_fill;
  stability_factor = p.stability_factor;
  camera_to_TCP = p.camera_to_TCP;
  model_directory = p.model_directory;
}

// The parameters are globals, so put them back after each test
class ConfigTest : public ::testing::Test {
protected:
  void SetUp() override { saved_ = currentParameters(); }
  void TearDown() override { setParameters(saved_); }

private:
  Parameters saved_;
};

}  // namespace

TEST_F(ConfigTest, LoadsTheValuesInTheFileAndKeepsTheRest) {
  const fs::path file = fs::temp_directory_path() / "pcl_pose_config_test.yaml";
  {
    std::ofstream out(file);
    out << "%YAML:1.0\n---\n"
           "leaf_size: 0.01\n"
           "min_cluster_size: 123\n"
           "model_directory: \"../somewhere/\"\n"
           "camera_to_TCP_x: 0.5\n"
           "stability_factor: 2\n";
  }
  const Parameters before = currentParameters();

  ASSERT_TRUE(loadConfig(file.string()));
  fs::remove(file);

  EXPECT_FLOAT_EQ(leaf_size, 0.01f);
  EXPECT_EQ(min_cluster_size, 123);
  EXPECT_EQ(model_directory, "../somewhere/");
  EXPECT_FLOAT_EQ(camera_to_TCP(0, 3), 0.5f);
  EXPECT_EQ(stability_factor, PoseUpdateStabilityFactor::Willing);
  // Not in the file, so unchanged
  EXPECT_FLOAT_EQ(camera_to_TCP(1, 3), before.camera_to_TCP(1, 3));
  EXPECT_FLOAT_EQ(cluster_tolerance, before.cluster_tolerance);
  EXPECT_EQ(cam_fps, before.cam_fps);
}

TEST_F(ConfigTest, TheStabilityFactorIsKeptInItsRange) {
  const fs::path file = fs::temp_directory_path() / "pcl_pose_config_range_test.yaml";
  {
    std::ofstream out(file);
    out << "%YAML:1.0\n---\nstability_factor: 9\n";
  }
  ASSERT_TRUE(loadConfig(file.string()));
  EXPECT_EQ(stability_factor, PoseUpdateStabilityFactor::VeryResistant);

  {
    std::ofstream out(file);
    out << "%YAML:1.0\n---\nstability_factor: -3\n";
  }
  ASSERT_TRUE(loadConfig(file.string()));
  EXPECT_EQ(stability_factor, PoseUpdateStabilityFactor::VeryWilling);
  fs::remove(file);
}

TEST_F(ConfigTest, AFileThatDoesNotExistCannotBeLoaded) {
  EXPECT_FALSE(loadConfig("this/file/does/not/exist.yaml"));
}

TEST_F(ConfigTest, TheDefaultFileMatchesTheDefaultsInTheCode) {
  // config/default.yaml documents every key with its default value, so loading it must not change anything
  const Parameters before = currentParameters();
  ASSERT_TRUE(loadConfig(std::string(POSE_SOURCE_DIR) + "/config/default.yaml"));
  const Parameters after = currentParameters();

  EXPECT_EQ(after.image_reduced_to_percentage, before.image_reduced_to_percentage);
  EXPECT_EQ(after.cam_res_width, before.cam_res_width);
  EXPECT_EQ(after.cam_res_height, before.cam_res_height);
  EXPECT_EQ(after.cam_fps, before.cam_fps);
  EXPECT_EQ(after.min_cluster_size, before.min_cluster_size);
  EXPECT_EQ(after.max_cluster_size, before.max_cluster_size);
  EXPECT_EQ(after.icp_max_iterations, before.icp_max_iterations);
  EXPECT_FLOAT_EQ(after.depth_filter_min_distance, before.depth_filter_min_distance);
  EXPECT_FLOAT_EQ(after.depth_filter_max_distance, before.depth_filter_max_distance);
  EXPECT_FLOAT_EQ(after.leaf_size, before.leaf_size);
  EXPECT_FLOAT_EQ(after.cluster_tolerance, before.cluster_tolerance);
  EXPECT_FLOAT_EQ(after.segment_distance_threshold, before.segment_distance_threshold);
  EXPECT_FLOAT_EQ(after.segment_probability, before.segment_probability);
  EXPECT_FLOAT_EQ(after.segment_radius_min, before.segment_radius_min);
  EXPECT_FLOAT_EQ(after.segment_radius_max, before.segment_radius_max);
  EXPECT_FLOAT_EQ(after.filter_cluster_to_x_percent, before.filter_cluster_to_x_percent);
  EXPECT_FLOAT_EQ(after.icp_transformation_epsilon, before.icp_transformation_epsilon);
  EXPECT_FLOAT_EQ(after.icp_euclidean_fitness_epsilon, before.icp_euclidean_fitness_epsilon);
  EXPECT_FLOAT_EQ(after.start_scale_distance, before.start_scale_distance);
  EXPECT_FLOAT_EQ(after.stop_scale_distance, before.stop_scale_distance);
  EXPECT_FLOAT_EQ(after.spray_collision_offset, before.spray_collision_offset);
  EXPECT_FLOAT_EQ(after.object_angle_calibration_offset, before.object_angle_calibration_offset);
  EXPECT_FLOAT_EQ(after.depth_filter_smooth_alpha, before.depth_filter_smooth_alpha);
  EXPECT_FLOAT_EQ(after.depth_filter_smooth_delta, before.depth_filter_smooth_delta);
  EXPECT_FLOAT_EQ(after.depth_filter_temporal_holes_fill, before.depth_filter_temporal_holes_fill);
  EXPECT_FLOAT_EQ(after.depth_filter_holes_fill, before.depth_filter_holes_fill);
  EXPECT_EQ(after.stability_factor, before.stability_factor);
  EXPECT_TRUE(after.camera_to_TCP.isApprox(before.camera_to_TCP, 1e-6f));
  EXPECT_EQ(after.model_directory, before.model_directory);
}
