#pragma once
#include <string>
#include <vector>

#include <Eigen/Dense>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

namespace pcl_pose {

// Returns the paths of all the object clouds in a library folder and its sub-folders, sorted by name. The paired
// _vfh.pcd signature files are not included.
std::vector<std::string> findLibraryClouds(const std::string& library_directory);

// Returns the centroid of the cloud (x, y, z, 1)
Eigen::Vector4f cloudCentroid(const pcl::PointCloud<pcl::PointXYZ>& cloud);

// Returns the VFH file that belongs to a library cloud: <name>.pcd -> <name>_vfh.pcd
std::string vfhFileForCloud(const std::string& cloud_file);

// Moves the cloud so that its centroid is at the origin, which is where the research report puts the coordinate
// system of each library cloud. Returns the centroid that the cloud was moved from.
Eigen::Vector4f centreOnCentroid(pcl::PointCloud<pcl::PointXYZ>& cloud);

}  // namespace pcl_pose
