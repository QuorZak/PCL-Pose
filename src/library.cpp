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
  Eigen::Vector4f centroid = Eigen::Vector4f::Zero(); // An empty cloud leaves it untouched
  compute3DCentroid(cloud, centroid);
  return centroid;
}

std::string vfhFileForCloud(const std::string& cloud_file) {
  const std::string extension = ".pcd";
  const std::string stem = cloud_file.substr(0, cloud_file.size() - extension.size());
  return stem + "_vfh" + extension;
}

std::string cloudFileForVfh(const std::string& vfh_file) {
  const std::string vfh_ending = "_vfh.pcd";
  if (vfh_file.size() < vfh_ending.size()
      || vfh_file.compare(vfh_file.size() - vfh_ending.size(), vfh_ending.size(), vfh_ending) != 0) {
    return vfh_file;
  }
  return vfh_file.substr(0, vfh_file.size() - vfh_ending.size()) + ".pcd";
}

bool isSprayBottleModel(const std::string& model_file) {
  return std::filesystem::path(model_file).filename().string().find("spray_bottle") != std::string::npos;
}

void transformCloudAndViewpoint(pcl::PointCloud<pcl::PointXYZ>& cloud, const Eigen::Matrix4f& transform) {
  pcl::transformPointCloud(cloud, cloud, transform);
  const Eigen::Matrix3f rotation = transform.block<3, 3>(0, 0);
  const Eigen::Vector3f origin = rotation * cloud.sensor_origin_.head<3>() + transform.block<3, 1>(0, 3);
  cloud.sensor_origin_.head<3>() = origin;
  cloud.sensor_orientation_ = (Eigen::Quaternionf(rotation) * cloud.sensor_orientation_).normalized();
}

Eigen::Vector4f centreOnCentroid(pcl::PointCloud<pcl::PointXYZ>& cloud) {
  const Eigen::Vector4f centroid = cloudCentroid(cloud);
  Eigen::Matrix4f transform = Eigen::Matrix4f::Identity();
  transform.block<3, 1>(0, 3) = -centroid.head<3>();
  transformCloudAndViewpoint(cloud, transform);
  return centroid;
}

bool viewpointIsLost(const pcl::PointCloud<pcl::PointXYZ>& cloud) {
  constexpr float minimum_camera_distance = 0.05f;
  const Eigen::Vector4f centroid = cloudCentroid(cloud);
  return (cloud.sensor_origin_.head<3>() - centroid.head<3>()).norm() < minimum_camera_distance;
}

}  // namespace pcl_pose
