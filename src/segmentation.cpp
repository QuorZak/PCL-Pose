#include "pcl_pose/segmentation.h"
#include "pcl_pose/camera.h"
#include "pcl_pose/parameters.h"

#include <iostream>

#include <pcl/ModelCoefficients.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/search/kdtree.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/segmentation/sac_segmentation.h>

namespace pcl_pose {

void filterAndSegmentPointCloud(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud,
  pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud_filtered, std::vector<pcl::PointIndices>& cluster_indices,
  const bool debugging) {

  // Create the filtering object: down-sample the dataset using a leaf size of 1cm
  pcl::VoxelGrid<pcl::PointXYZ> vg;
  vg.setInputCloud(cloud);
  vg.setLeafSize(leaf_size, leaf_size, leaf_size);
  vg.filter(*cloud_filtered);
  // std::cout << "PointCloud after filtering has: " << cloud_filtered->size() << " data points." << std::endl;

  // Create the segmentation object for the planar model and set all the parameters
  pcl::SACSegmentation<pcl::PointXYZ> seg;
  pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
  pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_plane(new pcl::PointCloud<pcl::PointXYZ>());
  seg.setOptimizeCoefficients(true);
  seg.setModelType(pcl::SACMODEL_PLANE);
  seg.setMethodType(pcl::SAC_RANSAC);
  seg.setMaxIterations(100); // 100 default
  seg.setDistanceThreshold(segment_distance_threshold); // 0.02 default

  // Additional parameters to improve sample quality
  seg.setProbability(segment_probability);
  seg.setRadiusLimits(segment_radius_min, segment_radius_max);

  int filter_count = 0;
  int nr_points = static_cast<int>(cloud_filtered->size());

  // Create the filtering object
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_f(new pcl::PointCloud<pcl::PointXYZ>);
  while (cloud_filtered->size () > filter_cluster_to_x_percent * nr_points) {
  // while (filter_count < 1) {
    // Segment the largest planar component from the remaining cloud
    seg.setInputCloud(cloud_filtered);
    seg.segment(*inliers, *coefficients);
    if (inliers->indices.empty()) {
      // No (more) planes: go on and look for the clusters in what is left. Returning here used to give no clusters at
      // all for a scene without a table or floor in view.
      if (debugging) {
        std::cout << "Could not estimate a planar model for the given dataset." << std::endl;
      }
      break;
    }

    // Extract the planar inliers from the input cloud
    pcl::ExtractIndices<pcl::PointXYZ> extract;
    extract.setInputCloud(cloud_filtered);
    extract.setIndices(inliers);
    extract.setNegative(false);

    // Get the points associated with the planar surface
    extract.filter(*cloud_plane);
    if (debugging) {
      std::cout << "PointCloud representing the planar component: " << cloud_plane->size() << " data points." << std::endl;
    }

    // Remove the planar inliers, extract the rest
    extract.setNegative(true);
    extract.filter(*cloud_f);
    *cloud_filtered = *cloud_f;

    filter_count++;
  }

  // Creating the KdTree object for the search method of the extraction
  pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>);
  tree->setInputCloud(cloud_filtered);

  pcl::EuclideanClusterExtraction<pcl::PointXYZ> ec;
  ec.setClusterTolerance(cluster_tolerance);
  ec.setMinClusterSize(min_cluster_size);
  ec.setMaxClusterSize(max_cluster_size);
  ec.setSearchMethod(tree);
  ec.setInputCloud(cloud_filtered);
  ec.extract(cluster_indices);

  if (debugging)
  {
    std::cout << "Finished trying to find clusters " << std::endl;
  }
}

void processPointCloud(const rs2::pipeline& pipeline,
  std::shared_ptr<pcl::PointCloud<pcl::PointXYZ>>& output_stream_cloud,
  const std::shared_ptr<std::vector<pcl::PointIndices>>& cluster_indices,
  rs2_vector* accel_data,
  const bool& debug, std::mutex& mtx, std::condition_variable& cv, bool& ready,
  const std::atomic<bool>& stop_requested)
{
  initialise_filters();
  rs2_vector latest_accel{};
  bool have_accel = false;
  while (!stop_requested) {
    auto frames = pipeline.wait_for_frames();

    // Keep the latest accelerometer reading. The accelerometer and the depth camera do not always arrive in the same
    // frameset, so this is done for every frameset.
    if (accel_data != nullptr) {
      if (const rs2::frame accel_frame = frames.first_or_default(RS2_STREAM_ACCEL)) {
        latest_accel = accel_frame.as<rs2::motion_frame>().get_motion_data();
        have_accel = true;
      }
    }

    const auto depth = frames.get_depth_frame();
    if (!depth) {
      continue; // This frameset only had motion data
    }
    if (accel_data != nullptr && !have_accel) {
      continue; // Wait until we know which way is up
    }
    rs2::depth_frame filtered_depth = apply_depth_post_processing_filters(depth);
    auto cloud = depthFrameToPointCloud(filtered_depth, true, debug);
    pcl::PointCloud<pcl::PointXYZ>::Ptr filtered_cloud_clusters(new pcl::PointCloud<pcl::PointXYZ>);
    std::vector<pcl::PointIndices> cluster_indices_local;
    filterAndSegmentPointCloud(cloud, filtered_cloud_clusters, cluster_indices_local, debug);

    if (cluster_indices_local.empty()) {
      if (debug) {
        std::cout << "No clusters found." << std::endl;
      }
      continue;
    }

    {
      std::lock_guard<std::mutex> lock(mtx);
      *output_stream_cloud = *filtered_cloud_clusters;
      *cluster_indices = cluster_indices_local;
      if (accel_data != nullptr) {
        *accel_data = latest_accel;
      }
      ready = true;
    }
    cv.notify_one();
  }
}

}  // namespace pcl_pose
