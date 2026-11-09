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

// The other way round: <name>_vfh.pcd -> <name>.pcd. A name that does not end in _vfh.pcd is returned as it is.
std::string cloudFileForVfh(const std::string& vfh_file);

// True if a library file is a spray bottle, which needs the spray_collision_offset: its file name (not the folders it
// is in) contains "spray_bottle", as the files saved by extraction --object spray_bottle_... do.
bool isSprayBottleModel(const std::string& model_file);

// Applies a rigid transform to the cloud and also to its viewpoint (sensor_origin_ and sensor_orientation_, where the
// camera was), so that the VFH signature of the moved cloud stays the same as the one the camera would see.
void transformCloudAndViewpoint(pcl::PointCloud<pcl::PointXYZ>& cloud, const Eigen::Matrix4f& transform);

// Moves the cloud (and its viewpoint) so that its centroid is at the origin, which is where the research report puts
// the coordinate system of each library cloud. Returns the centroid that the cloud was moved from.
Eigen::Vector4f centreOnCentroid(pcl::PointCloud<pcl::PointXYZ>& cloud);

// True if the cloud's viewpoint can not be right: it is within 5 cm of the cloud's centroid, which the camera never
// is (its minimum range is about 0.25 m). This is what a cloud that was centred without moving its viewpoint looks like.
bool viewpointIsLost(const pcl::PointCloud<pcl::PointXYZ>& cloud);

}  // namespace pcl_pose
