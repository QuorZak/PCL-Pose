#include "pose_estimation.h"

int main() {
  const std::string test_name = "test";
  const std::string output_folder = "../lab_data/test/";

  // Initialise the Realsense pipeline
  rs2::pipeline pipe;
  rs2::config config;
  config.enable_stream(RS2_STREAM_DEPTH, cam_res_width, cam_res_height, RS2_FORMAT_Z16, cam_fps); // Use global parameters
  pipe.start(config);

  // Camera warmup - dropping several first frames to let auto-exposure stabilize
  for (int i = 0; i < 100; i++) {
    // Wait for all configured streams to produce a frame
    auto frames = pipe.wait_for_frames(); // purposely does nothing with the frames
  }

  // Capture a frame
  rs2::frameset frames = pipe.wait_for_frames();
  rs2::frame depth = frames.get_depth_frame();
  depth = apply_post_processing_filters(depth);

  // Convert the depth frame to a PCL point cloud
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
  cloud = depthFrameToPointCloud(depth, true);

  pipe.stop();
  std::cout << "PointCloud captured from Realsense camera has: " << cloud->size() << " data points." << std::endl;

  // Call the extracted function
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_filtered(new pcl::PointCloud<pcl::PointXYZ>);
  std::vector<pcl::PointIndices> cluster_indices;
  filterAndSegmentPointCloud(cloud, cloud_filtered, cluster_indices);

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