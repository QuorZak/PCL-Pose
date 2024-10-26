#include <pcl/ModelCoefficients.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/search/kdtree.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/segmentation/extract_clusters.h>
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

  while (true) {
    std::cout << "Ready to capture point cloud. " << std::endl;
    std::cout << "Press any key when ready, or Q to quit" << std::endl;

    // Wait for the user to press a key
    char key = '';
    while (key == '')
    {
      key = cv::waitKey(0);
      if (key == 'Q' || key == 'q') {
        break;
      }
    }

    // Initialize the Realsense pipeline
    rs2::pipeline pipe;
    pipe.start();

    // Camera warmup - dropping several first frames to let auto-exposure stabilize
    for (int i = 0; i < 100; i++) {
      // Wait for all configured streams to produce a frame
      auto frames = pipe.wait_for_frames(); // purposely does nothing with the frames
    }

    // Capture a frame
    rs2::frameset frames = pipe.wait_for_frames();
    rs2::frame depth = frames.get_depth_frame();

    // Convert the depth frame to a PCL point cloud
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>), cloud_f(new pcl::PointCloud<pcl::PointXYZ>);
    cloud = depthFrameToPointCloud(depth, true, true);

    pipe.stop();
    std::cout << "PointCloud captured from Realsense camera has: " << cloud->size() << " data points." << std::endl;

    // Create the filtering object: downsample the dataset using a leaf size of 1cm
    pcl::VoxelGrid<pcl::PointXYZ> vg;
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_filtered (new pcl::PointCloud<pcl::PointXYZ>);
    vg.setInputCloud (cloud);
    vg.setLeafSize (0.01f, 0.01f, 0.01f); // 0.01f default
    vg.filter (*cloud_filtered);
    std::cout << "PointCloud after filtering has: " << cloud_filtered->size ()  << " data points." << std::endl; //*

    // Create the segmentation object for the planar model and set all the parameters
    pcl::SACSegmentation<pcl::PointXYZ> seg;
    pcl::PointIndices::Ptr inliers (new pcl::PointIndices);
    pcl::ModelCoefficients::Ptr coefficients (new pcl::ModelCoefficients);
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_plane (new pcl::PointCloud<pcl::PointXYZ> ());
    pcl::PCDWriter writer;
    seg.setOptimizeCoefficients (true);
    seg.setModelType (pcl::SACMODEL_PLANE);
    seg.setMethodType (pcl::SAC_RANSAC);
    seg.setMaxIterations (100); // 100 default
    seg.setDistanceThreshold (0.02); // 0.02 default

    int nr_points = static_cast<int>(cloud_filtered->size());
    int filter_count = 1;
    // while (cloud_filtered->size () > 0.3 * nr_points) // 0.3 default
    while (filter_count < 1)
    {
      // Segment the largest planar component from the remaining cloud
      seg.setInputCloud (cloud_filtered);
      seg.segment (*inliers, *coefficients);
      if (inliers->indices.empty())
      {
        std::cout << "Could not estimate a planar model for the given dataset." << std::endl;
        return 1;
      }

      // Extract the planar inliers from the input cloud
      pcl::ExtractIndices<pcl::PointXYZ> extract;
      extract.setInputCloud (cloud_filtered);
      extract.setIndices (inliers);
      extract.setNegative (false);

      // Get the points associated with the planar surface
      extract.filter (*cloud_plane);
      std::cout << "PointCloud representing the planar component: " << cloud_plane->size () << " data points." << std::endl;

      // Remove the planar inliers, extract the rest
      extract.setNegative (true);
      extract.filter (*cloud_f);
      *cloud_filtered = *cloud_f;

      filter_count++;
    }

    // Creating the KdTree object for the search method of the extraction
    pcl::search::KdTree<pcl::PointXYZ>::Ptr tree (new pcl::search::KdTree<pcl::PointXYZ>);
    tree->setInputCloud (cloud_filtered);

    std::vector<pcl::PointIndices> cluster_indices;
    pcl::EuclideanClusterExtraction<pcl::PointXYZ> ec;
    ec.setClusterTolerance (0.02); // 2cm == 0.02 default
    ec.setMinClusterSize (100); // 100 default
    ec.setMaxClusterSize (25000); // 25000 default
    ec.setSearchMethod (tree);
    ec.setInputCloud (cloud_filtered);
    ec.extract (cluster_indices);

    /*// Clear out all the files in the output folder from the previous run
    const std::string clear_command = "rm -f " + output_folder + object_name + *.pcd";
    if (const int result = std::system(clear_command.c_str()); result != 0) {
      std::cerr << "Failed to execute rm command" << std::endl;
    } else {
      std::cout << "Cleared out all the files in the output folder" << std::endl;
    }*/

    // If no clusters are found, output a message
    if (!cluster_indices.empty())
    {
      int j = 0;
      std::vector<std::string> created_files;
      for (const auto& cluster : cluster_indices)
      {
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_cluster (new pcl::PointCloud<pcl::PointXYZ>);
        for (const auto& idx : cluster.indices) {
          cloud_cluster->push_back((*cloud_filtered)[idx]);
        } //*
        cloud_cluster->width = cloud_cluster->size ();
        cloud_cluster->height = 1;
        cloud_cluster->is_dense = true;

        // Save the point cloud
        std::cout << "PointCloud representing the Cluster: " << cloud_cluster->size () << " data points." << std::endl;
        std::stringstream ss;
        ss << std::setw(4) << std::setfill('0') << file_num++;
        std::string filename = output_folder + object_name + "_" + ss.str () + ".pcd";
        writer.write<pcl::PointXYZ> (filename, *cloud_cluster, false);
        created_files.push_back(filename);

        // Get the VFH signature of the cluster then save
        pcl::PointCloud<pcl::VFHSignature308> signature;
        estimate_VFH(cloud_cluster, signature);
        std::string vfh_filename = output_folder + object_name + "_" + ss.str ()+ "_vfh.pcd";
        pcl::io::savePCDFile(vfh_filename, signature);
        created_files.push_back(vfh_filename);

        j++;
      }

      const std::string show_command = "pcl_viewer " + output_folder += object_name + "_*.pcd";
      if (const int result = std::system(show_command.c_str()); result != 0) {
        std::cerr << "Failed to execute pcl_viewer command" << std::endl;
        return -1;
      }

      std::cout << "Keep the result? y/[n]" << std::endl;
      char choice = cv::waitKey(0);
      if (choice == 'y' || choice == 'Y') {
        continue;
      } else if (choice == 'n' || choice == 'N') {
        for (const auto& file : created_files) {
          std::remove(file.c_str());
        }
        std::cout << "Files deleted." << std::endl;
      } else {
        break;
      }
    } else {
      std::cout << "No clusters found." << std::endl;
      std::cout << "Do you want to continue?" << std::endl;
      char choice = cv::waitKey(0);
      if (choice != 'y' && choice != 'Y') {
        break;
      }
    }
  }

  return 0;
}