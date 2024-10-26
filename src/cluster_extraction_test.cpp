#include <pcl/ModelCoefficients.h>
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/search/kdtree.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/segmentation/extract_clusters.h>
#include <iomanip> // for setw, setfill

# include <librealsense2/rs.hpp> // Include RealSense Cross Platform API

#include "pose_estimation.h"

int main () {
  const std::string output_folder = "../lab_data/test/";
  const std::string test_name = "test";

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

  // Implement a depth threshold filter
  rs2::threshold_filter threshold_filter;
  threshold_filter.set_option(RS2_OPTION_MIN_DISTANCE, depth_filter_min_distance);
  threshold_filter.set_option(RS2_OPTION_MAX_DISTANCE, depth_filter_max_distance);
  depth = threshold_filter.process(depth);

  // Convert the depth frame to a PCL point cloud
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>), cloud_f(new pcl::PointCloud<pcl::PointXYZ>);
  rs2::pointcloud rs_cloud;
  rs2::points points = rs_cloud.calculate(depth);
  auto stream_profile = points.get_profile().as<rs2::video_stream_profile>();
  const int width = stream_profile.width();
  const int height = stream_profile.height();
  cloud->width = width;
  cloud->height = height;
  cloud->is_dense = false;
  cloud->points.resize(points.size());

  // Filter the point cloud then extract the points in the center
  // We will want to crop the image to focus only on the center
  const auto [x_start, x_stop, y_start, y_stop]
              = get_crop_points(width, height, image_reduced_to_percentage);

  // Extract the points
  auto vertices = points.get_vertices();
  int i = 0;
  for (int y = y_start; y < y_stop; y++) {
    for (int x = x_start; x < x_stop; x++, i++) {
      cloud->points[i].x = vertices[i].x;
      cloud->points[i].y = -vertices[i].y;
      cloud->points[i].z = -vertices[i].z;
    }
  }
  // set the size of the point cloud
  cloud->width = x_stop - x_start;
  cloud->height = y_stop - y_start;

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
      break;
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

  // Clear out all the files in the output folder from the previous run
  const std::string clear_command = "rm -f " + output_folder + test_name + "_*.pcd";
  if (const int result = std::system(clear_command.c_str()); result != 0) {
    std::cerr << "Failed to execute rm command" << std::endl;
  } else {
    std::cout << "Cleared out all the files in the output folder" << std::endl;
  }

  // If no clusters are found, output a message
  if (cluster_indices.empty())
  {
    std::cout << "No clusters found." << std::endl;
    return (0);
  }
  int j = 0;
  for (const auto& cluster : cluster_indices)
  {
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_cluster (new pcl::PointCloud<pcl::PointXYZ>);
    for (const auto& idx : cluster.indices) {
      cloud_cluster->push_back((*cloud_filtered)[idx]);
    } //*
    cloud_cluster->width = cloud_cluster->size ();
    cloud_cluster->height = 1;
    cloud_cluster->is_dense = true;

    std::cout << "PointCloud representing the Cluster: " << cloud_cluster->size () << " data points." << std::endl;
    std::stringstream ss;
    ss << std::setw(4) << std::setfill('0') << j;
    writer.write<pcl::PointXYZ> (output_folder + test_name +"_" + ss.str () + ".pcd", *cloud_cluster, false);
    j++;
  }

  const std::string show_command = "pcl_viewer " + output_folder + test_name + "_*.pcd";
  if (const int result = std::system(show_command.c_str()); result != 0) {
    std::cerr << "Failed to execute pcl_viewer command" << std::endl;
  }

  return (0);
}
