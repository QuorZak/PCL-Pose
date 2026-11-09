#pragma once
#include <Eigen/Dense>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

namespace pcl_pose {

// How the part of the object that the robot aims at is chosen
struct TargetBand {
  // Lower and upper edge of the band in y. Both are measured from the middle of the object, as a fraction of half the
  // object's height. In the robot's TCP frame y points down, so a positive fraction is the lower part of the object.
  float start = 0.30f;
  float stop = 0.70f;
  // The band is this fraction of the object's width either side of the object's mean x
  float width = 0.25f;
};

// Finds the point the robot should aim for: the point closest to the camera (smallest z) inside a band around the
// middle-lower part of the object. The object's points must be in the robot's TCP frame, in metres.
// Returns false if the cloud is empty or the band contains no points.
// If band is not a nullptr, it is filled with the points of the band.
bool findClosestPointInBand(const pcl::PointCloud<pcl::PointXYZ>& object, Eigen::Vector3f& closest_point,
  pcl::PointCloud<pcl::PointXYZ>* band = nullptr, const TargetBand& settings = TargetBand());

}  // namespace pcl_pose
