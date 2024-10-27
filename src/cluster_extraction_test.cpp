#include "pose_estimation.h"
#include <pcl/common/transforms.h>
#include <librealsense2/rs.hpp> // Include RealSense Cross Platform API
#include <pcl/common/centroid.h>

int main() {
  const std::string test_name = "test";
  const std::string output_folder = "../lab_data/test/";
  const float object_facing_angle = 0.0f; // Change this each capture

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

  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
  cloud = depthFrameToPointCloud(depth, true);

  pipe.stop();
  std::cout << "PointCloud captured from Realsense camera has: " << cloud->size() << " data points." << std::endl;

  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_filtered(new pcl::PointCloud<pcl::PointXYZ>);
  std::vector<pcl::PointIndices> cluster_indices;
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

    Eigen::Vector4f centroid;
    pcl::compute3DCentroid(*cloud_cluster, centroid);

    Eigen::Affine3f transform = Eigen::Affine3f::Identity();
    transform.translation() << -centroid[0], -centroid[1], -centroid[2];
    pcl::transformPointCloud(*cloud_cluster, *cloud_cluster, transform);

    // TODO: Figure out this object rotation a bit better
    Eigen::Matrix3f object_rotation; // Some - and + values to get 0 to be approximately directly away from the camera
    object_rotation = Eigen::AngleAxisf(M_PI * (object_facing_angle+90) / -180.0f, Eigen::Vector3f::UnitY());

    Eigen::Vector3f gravity_direction(accel_data.x, accel_data.y, accel_data.z);
    gravity_direction.normalize();

    // Calculate the angle between the y-axis and the gravity direction
    Eigen::Vector3f y_axis = Eigen::Vector3f::UnitY();
    float angle = acos(y_axis.dot(gravity_direction));

    // Subtract the angle from the gravity direction
    Eigen::Quaternionf rotation_quat = Eigen::Quaternionf(Eigen::AngleAxisf(-angle, y_axis.cross(gravity_direction).normalized()));
    Eigen::Matrix3f align_rotation = rotation_quat.toRotationMatrix();

    // Apply the inverse of the resulting transformation to the point cloud
    Eigen::Matrix4f new_pose = Eigen::Matrix4f::Identity();
    new_pose.block<3, 3>(0, 0) = -align_rotation;

    // Now that the y direction is set, we can rotate the object to face the camera
    new_pose.block<3, 3>(0, 0) = object_rotation * new_pose.block<3, 3>(0, 0);

    // The x-axis can be set to the cross product of the y-axis and z-axis
    new_pose.block<3, 1>(0, 0) = new_pose.block<3, 1>(0, 1).cross(new_pose.block<3, 1>(0, 2));

    pcl::transformPointCloud(*cloud_cluster, *cloud_cluster, new_pose);

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