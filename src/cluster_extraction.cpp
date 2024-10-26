#include <pcl/ModelCoefficients.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <iomanip> // for setw, setfill
#include <iostream>
#include <cstdio> // for std::remove
#include <filesystem>

#include "pose_estimation.h"

int main () {
  const std::string object_name = "mustard_small";
  const std::string output_folder = "../lab_data/" + object_name + "/";

  // Get the highest file number
  int file_num = 0;
  for (const auto& entry : std::filesystem::directory_iterator(output_folder)) {
    std::string filename = entry.path().filename().string();
    if (filename.find(object_name) != std::string::npos && filename.find(".pcd") != std::string::npos) {
      int num = std::stoi(filename.substr(object_name.size() + 1, 4));
      if (num > file_num) {
        file_num = num;
      }
    }
  }
  file_num += 1;

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

  while (true) {
    std::cout << "Ready to capture point cloud. " << std::endl;
    std::cout << "Press any key when ready, or Q to quit" << std::endl;

    char key;
    std::cin >> key;
    if (key == 'Q' || key == 'q') {
      break;
    }

    // Capture a frame
    rs2::frameset frames = pipe.wait_for_frames();
    rs2::frame depth = frames.get_depth_frame();

    // Convert the depth frame to a PCL point cloud
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>), cloud_f(new pcl::PointCloud<pcl::PointXYZ>);
    cloud = depthFrameToPointCloud(depth, true, true);

    std::cout << "PointCloud captured from Realsense camera has: " << cloud->size() << " data points." << std::endl;

    // Call the extracted function
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_filtered(new pcl::PointCloud<pcl::PointXYZ>);
    std::vector<pcl::PointIndices> cluster_indices;
    filterAndSegmentPointCloud(cloud, cloud_filtered, cloud_f, cluster_indices);

    // If no clusters are found, output a message
    if (!cluster_indices.empty()) {
      int j = 0;
      std::vector<std::string> created_files;
      pcl::PCDWriter writer;
      for (const auto& cluster : cluster_indices) {
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_cluster(new pcl::PointCloud<pcl::PointXYZ>);
        for (const auto& idx : cluster.indices) {
          cloud_cluster->push_back((*cloud_filtered)[idx]);
        }
        cloud_cluster->width = cloud_cluster->size();
        cloud_cluster->height = 1;
        cloud_cluster->is_dense = true;

        // Save the point cloud
        std::cout << "PointCloud representing the Cluster: " << cloud_cluster->size() << " data points." << std::endl;
        std::stringstream ss;
        ss << std::setw(4) << std::setfill('0') << file_num++;
        std::string filename = output_folder + object_name + "_" + ss.str() + ".pcd";
        writer.write<pcl::PointXYZ>(filename, *cloud_cluster, false);
        created_files.push_back(filename);

        // Get the VFH signature of the cluster then save
        pcl::PointCloud<pcl::VFHSignature308> signature;
        estimate_VFH(cloud_cluster, signature);
        std::string vfh_filename = output_folder + object_name + "_" + ss.str() + "_vfh.pcd";
        pcl::io::savePCDFile(vfh_filename, signature);
        created_files.push_back(vfh_filename);

        j++;
      }

      // Display the results to review
      showPointClouds(created_files);

      std::cout << "Keep the result? y/[n]" << std::endl;
      char choice;
      std::cin >> choice;
      if (choice == 'y' || choice == 'Y') {
        std::cout << "Files saved." << std::endl;
        continue;
      } else {
        for (const auto& file : created_files) {
          std::remove(file.c_str());
        }
        std::cout << "Files deleted." << std::endl;
      }
    } else {
      std::cout << "No clusters found." << std::endl;
      std::cout << "Do you want to continue? y/[n]" << std::endl;
      char choice;
      std::cin >> choice;
      if (choice != 'y' && choice != 'Y') {
        break;
      }
    }
  }
  pipe.stop();
  return 0;
}