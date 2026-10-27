#include "pcl_pose/library.h"

#include <algorithm>
#include <filesystem>

#include <pcl/common/centroid.h>
#include <pcl/common/transforms.h>

namespace pcl_pose {

std::vector<std::string> findLibraryClouds(const std::string& library_directory) {
  namespace fs = std::filesystem;
  std::vector<std::string> clouds;
  if (!fs::is_directory(library_directory)) {
    return clouds;
  }
  for (const auto& entry : fs::recursive_directory_iterator(library_directory)) {
    if (!entry.is_regular_file() || entry.path().extension() != ".pcd") {
      continue;
    }
    const std::string name = entry.path().filename().string();
    const std::string vfh_ending = "_vfh.pcd";
    const bool is_vfh_file = name.size() >= vfh_ending.size()
      && name.compare(name.size() - vfh_ending.size(), vfh_ending.size(), vfh_ending) == 0;
    if (!is_vfh_file) {
      clouds.push_back(entry.path().string());
    }
  }
  std::sort(clouds.begin(), clouds.end());
  return clouds;
}

Eigen::Vector4f cloudCentroid(const pcl::PointCloud<pcl::PointXYZ>& cloud) {
  Eigen::Vector4f centroid;
  compute3DCentroid(cloud, centroid);
  return centroid;
}

std::string vfhFileForCloud(const std::string& cloud_file) {
  const std::string extension = ".pcd";
  const std::string stem = cloud_file.substr(0, cloud_file.size() - extension.size());
  return stem + "_vfh" + extension;
}

Eigen::Vector4f centreOnCentroid(pcl::PointCloud<pcl::PointXYZ>& cloud) {
  const Eigen::Vector4f centroid = cloudCentroid(cloud);
  Eigen::Affine3f transform = Eigen::Affine3f::Identity();
  transform.translation() << -centroid[0], -centroid[1], -centroid[2];
  transformPointCloud(cloud, cloud, transform);
  return centroid;
}

}  // namespace pcl_pose
