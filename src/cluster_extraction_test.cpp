#include "pose_estimation.h"
#include <pcl/common/transforms.h>
#include <librealsense2/rs.hpp> // Include RealSense Cross Platform API

int main() {
  const std::string test_name = "test";
  const std::string output_folder = "../lab_data/test/";
  // Define the angle (in degrees) that the object is facing
  float object_facing_angle = 45.0f; // Change this each capture

  // Initialise the Realsense pipeline
  rs2::pipeline pipe;
  rs2::config config;
  config.enable_stream(RS2_STREAM_DEPTH, cam_res_width, cam_res_height, RS2_FORMAT_Z16, cam_fps); // Use global parameters
  config.enable_stream(RS2_STREAM_ACCEL, RS2_FORMAT_MOTION_XYZ32F);
  config.enable_stream(RS2_STREAM_GYRO, RS2_FORMAT_MOTION_XYZ32F);
  pipe.start(config);

  // Initialise the filters which will be applied to the depth frame
  initialise_filters();

  // Camera warmup - dropping several first frames to let auto-exposure stabilize
  for (int i = 0; i < 100; i++) {
    // Wait for all configured streams to produce a frame
    auto frames = pipe.wait_for_frames(); // purposely does nothing with the frames
  }

  // Capture a frame
  rs2::frameset frames = pipe.wait_for_frames();
  rs2::frame depth = frames.get_depth_frame();
  depth = apply_post_processing_filters(depth);

  // Get gyroscopic data
  rs2::motion_frame gyro_frame = frames.first_or_default(RS2_STREAM_GYRO);
  rs2_vector gyro_data = gyro_frame.get_motion_data();

  // Convert the depth frame to a PCL point cloud
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
  cloud = depthFrameToPointCloud(depth, true);

  pipe.stop();
  std::cout << "PointCloud captured from Realsense camera has: " << cloud->size() << " data points." << std::endl;

  // Call the extracted function
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_filtered(new pcl::PointCloud<pcl::PointXYZ>);
  std::vector<pcl::PointIndices> cluster_indices;
  filterAndSegmentPointCloud(cloud, cloud_filtered, cluster_indices, true);

  // Clear out all the files in the output folder from the previous run
  const std::string clear_command = "rm -f " + output_folder + test_name + "_*.pcd";
  if (const int result = std::system(clear_command.c_str()); result != 0) {
    std::cerr << "Failed to execute rm command" << std::endl;
  } else {
    std::cout << "Cleared out all the files in the output folder" << std::endl;
  }

  // If no clusters are found, output a message
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

    // Define the forward direction based on the provided angle
    Eigen::Matrix3f object_rotation;
    object_rotation = Eigen::AngleAxisf(object_facing_angle * M_PI / 180.0f, Eigen::Vector3f::UnitY());

    // Create a rotation matrix from the gyroscopic data
    Eigen::Matrix3f rotation_matrix;
    rotation_matrix = Eigen::AngleAxisf(gyro_data.z, Eigen::Vector3f::UnitZ()) *
                      Eigen::AngleAxisf(gyro_data.y, Eigen::Vector3f::UnitY()) *
                      Eigen::AngleAxisf(gyro_data.x, Eigen::Vector3f::UnitX());

    // Align the y-direction of the object cloud with the y-direction in the real world (gravity)
    Eigen::Vector3f gravity_direction(0.0f, 1.0f, 0.0f);
    Eigen::Vector3f object_y_direction = rotation_matrix * gravity_direction;

    // Combine the rotations
    Eigen::Matrix3f combined_rotation = object_rotation * rotation_matrix;

    // Set the new pose
    Eigen::Matrix4f new_pose = Eigen::Matrix4f::Identity();
    new_pose.block<3, 3>(0, 0) = combined_rotation;
    new_pose.block<3, 1>(0, 1) = object_y_direction.normalized();
    new_pose.block<3, 1>(0, 0) = object_y_direction.cross(combined_rotation.col(2)).normalized();
    // Transform the point cloud using the new pose
    transformPointCloud(*cloud_cluster, *cloud_cluster, new_pose);

    std::cout << "PointCloud representing the Cluster: " << cloud_cluster->size() << " data points." << std::endl;
    std::stringstream ss;
    ss << std::setw(4) << std::setfill('0') << j;
    writer.write<pcl::PointXYZ>(output_folder + test_name + "_" + ss.str() + ".pcd", *cloud_cluster, false);
    j++;
  }

  const std::string files_to_show_pattern = output_folder + test_name + "_*.pcd";
  std::vector<std::string> files_to_show = globFiles(files_to_show_pattern);

  // Display the results to review
  showPointClouds(files_to_show);

  return 0;
}