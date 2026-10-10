#include <pcl/ModelCoefficients.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <iomanip> // for setw, setfill
#include <iostream>
#include <filesystem>
#include <regex>

#include <pose_estimation.h>

using namespace pcl_pose;

int main() {
  const std::string test_name = "spray_bottle_tall";
  const std::string output_folder = "../lab_data/test/";
  //const std::string output_folder = "../lab_data/" + test_name + "/";

  // Get the highest file number
  int file_num = 0;
  for (const auto& entry : std::filesystem::directory_iterator(output_folder)) {
    std::string filename = entry.path().filename().string();
    if (filename.find(test_name) != std::string::npos && filename.find(".pcd") != std::string::npos) {
      int num = std::stoi(filename.substr(test_name.size() + 1, 4));
      if (num > file_num) {
        file_num = num;
      }
    }
  }
  file_num += 1;

  // Declare any variables
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);

  rs2::pipeline pipe;
  rs2::config config;
  config.enable_stream(RS2_STREAM_DEPTH, cam_res_width, cam_res_height, RS2_FORMAT_Z16, cam_fps); // Use global parameters
  pipe.start(config);

  // Initialise the filters which will be applied to the depth frame
  initialise_filters();

  // Camera warmup - dropping several first frames to let auto-exposure stabilize
  for (int i = 0; i < 100; i++) {
    auto frames = pipe.wait_for_frames(); // purposely does nothing with the frames
  }

  rs2::frameset frames = pipe.wait_for_frames();
  rs2::frame depth = frames.get_depth_frame();
  depth = apply_depth_post_processing_filters(depth);

  // The clouds saved by this program are left as seen by the camera (no re-orientation or centring)
  cloud = depthFrameToPointCloud(depth, true, true);

  pipe.stop();

  // Filter and segment the point cloud
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_filtered(new pcl::PointCloud<pcl::PointXYZ>);
  std::vector<pcl::PointIndices> cluster_indices;
  filterAndSegmentPointCloud(cloud, cloud_filtered, cluster_indices, true);

  // Clear out the previous captures of this object from the output folder
  for (const auto& old_file : listPcdFiles(output_folder, test_name + "_")) {
    std::error_code error;
    std::filesystem::remove(old_file, error);
    if (error) {
      std::cerr << "Failed to remove " << old_file << ": " << error.message() << std::endl;
    }
  }
  std::cout << "Cleared out the previous captures from the output folder" << std::endl;

  if (cluster_indices.empty()) {
    std::cout << "No clusters found." << std::endl;
    return 1;
  }

  // Save the point clouds
  pcl::PCDWriter writer;
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
    ss << std::setw(4) << std::setfill('0') << file_num++;
    writer.write<pcl::PointXYZ>(output_folder + test_name + "_" + ss.str() + ".pcd", *cloud_cluster, false);

    // Get the VFH signature of the cluster then save
    pcl::PointCloud<pcl::VFHSignature308> signature;
    estimate_VFH(cloud_cluster, signature);
    std::string vfh_filename = output_folder + test_name + "_" + ss.str() + "_vfh.pcd";
    pcl::io::savePCDFile(vfh_filename, signature);
  }

  // Show the clouds that were just saved (the _vfh files belong to the same clouds, so skip them)
  showPointClouds(listPcdFiles(output_folder, test_name + "_"), true);

  return 0;
}