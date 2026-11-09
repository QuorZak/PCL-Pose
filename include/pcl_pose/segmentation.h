#pragma once
#include <atomic>
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

// Meant to run on its own thread until stop_requested is set: captures depth frames, turns them into a point cloud and
// extracts the clusters. The latest result is copied into output_stream_cloud and cluster_indices under the mutex,
// and ready is set and the condition variable notified.
//
// This function is the only one that should read frames from the pipeline. If accel_data is not a nullptr, the
// pipeline must have the accelerometer stream enabled, and the latest accelerometer reading is written to
// accel_data (under the same mutex, together with each result), so the caller does not have to read frames itself.
void processPointCloud(const rs2::pipeline& pipeline,
  std::shared_ptr<pcl::PointCloud<pcl::PointXYZ>>& output_stream_cloud,
  const std::shared_ptr<std::vector<pcl::PointIndices>>& cluster_indices,
  rs2_vector* accel_data,
  const bool& debug, std::mutex& mtx, std::condition_variable& cv, bool& ready,
  const std::atomic<bool>& stop_requested);

}  // namespace pcl_pose
