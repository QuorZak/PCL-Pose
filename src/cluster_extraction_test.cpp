#include "pose_estimation.h"
#include <pcl/common/transforms.h>
#include <librealsense2/rs.hpp> // Include RealSense Cross Platform API

int main() {
  const std::string test_name = "test";
  const std::string output_folder = "../lab_data/test/";
  const float object_facing_angle = 0.0f; // Change this each capture
  const float calibration_angle_offset = -40.0f; // Set this if your camera is not quite aligned

  // Declare any variables - put here just to align more with the cluster_extraction.cpp
  std::unique_ptr<Eigen::Matrix4f> object_pose(new Eigen::Matrix4f());
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_filtered(new pcl::PointCloud<pcl::PointXYZ>);
  std::vector<pcl::PointIndices> cluster_indices;

  rs2::pipeline pipe;
  rs2::config config;
  config.enable_stream(RS2_STREAM_DEPTH, cam_res_width, cam_res_height, RS2_FORMAT_Z16, cam_fps); // Use global parameters
  config.enable_stream(RS2_STREAM_ACCEL, RS2_FORMAT_MOTION_XYZ32F);
  pipe.start(config);

  initialise_filters();

  for (int i = 0; i < 100; i++) {
    auto frames = pipe.wait_for_frames(); // purposely does nothing with the frames
  }

  rs2::frameset frames = pipe.wait_for_frames();
  rs2::frame depth = frames.get_depth_frame();
  depth = apply_post_processing_filters(depth);

  rs2::motion_frame accel_frame = frames.first_or_default(RS2_STREAM_ACCEL);
  rs2_vector accel_data = accel_frame.get_motion_data();

  cloud = depthFrameToPointCloud(depth, true);

  pipe.stop();
  std::cout << "PointCloud captured from Realsense camera has: " << cloud->size() << " data points." << std::endl;

  filterAndSegmentPointCloud(cloud, cloud_filtered, cluster_indices, true);

  const std::string clear_command = "rm -f " + output_folder + test_name + "_*.pcd";
  if (const int result = std::system(clear_command.c_str()); result != 0) {
    std::cerr << "Failed to execute rm command" << std::endl;
  } else {
    std::cout << "Cleared out all the files in the output folder" << std::endl;
  }

  if (cluster_indices.empty()) {
    std::cout << "No clusters found." << std::endl;
    return 1;
  }

  pcl::PCDWriter writer;
  int j = 0;
  for (const auto& cluster : cluster_indices) {
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_cluster(new pcl::PointCloud<pcl::PointXYZ>);
    for (const auto& idx : cluster.indices) {
      cloud_cluster->push_back((*cloud_filtered)[idx]);
    }
    cloud_cluster->width = cloud_cluster->size();
    cloud_cluster->height = 1;
    cloud_cluster->is_dense = true;

    // Get the pose of the object
    getPointCloudOriginAndAxes(cloud_cluster, *object_pose, object_facing_angle, calibration_angle_offset, accel_data);

    // Transform the point cloud using the new pose
    transformPointCloud(*cloud_cluster, *cloud_cluster, *object_pose);

    std::cout << "PointCloud representing the Cluster: " << cloud_cluster->size() << " data points." << std::endl;
    std::stringstream ss;
    ss << std::setw(4) << std::setfill('0') << j;
    writer.write<pcl::PointXYZ>(output_folder + test_name + "_" + ss.str() + ".pcd", *cloud_cluster, false);
    j++;
  }

  const std::string files_to_show_pattern = output_folder + test_name + "_*.pcd";
  std::vector<std::string> files_to_show = globFiles(files_to_show_pattern);

  showPointClouds(files_to_show);

  return 0;
}