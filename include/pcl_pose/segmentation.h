#pragma once
#include <condition_variable>
#include <memory>
#include <mutex>
#include <vector>

#include <librealsense2/rs.hpp>

#include <pcl/PointIndices.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

namespace pcl_pose {

// This is the Cluster Extraction code that is common to both the test and the main code
// It down-samples the cloud, removes the planes (floor, table) and extracts the clusters from what is left
void filterAndSegmentPointCloud(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud,
  pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud_filtered, std::vector<pcl::PointIndices>& cluster_indices,
  bool debugging = false);

// Runs forever on its own thread: captures depth frames, turns them into a point cloud and extracts the clusters.
// The latest result is copied into output_stream_cloud and cluster_indices under the mutex, and ready is set
// and the condition variable notified.
[[noreturn]] void processPointCloud(const rs2::pipeline& pipeline,
  std::shared_ptr<pcl::PointCloud<pcl::PointXYZ>>& output_stream_cloud,
  const std::shared_ptr<std::vector<pcl::PointIndices>>& cluster_indices,
  const bool& debug, std::mutex& mtx, std::condition_variable& cv, bool& ready);

}  // namespace pcl_pose
