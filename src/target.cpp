#include "pcl_pose/target.h"

#include <algorithm>
#include <cstdint>
#include <limits>

namespace pcl_pose {

bool findClosestPointInBand(const pcl::PointCloud<pcl::PointXYZ>& object, Eigen::Vector3f& closest_point,
  pcl::PointCloud<pcl::PointXYZ>* band, const TargetBand& settings) {
  if (band != nullptr) {
    band->clear();
  }
  if (object.empty()) {
    return false;
  }

  // Get the middle (y) of the object. lowest() is the starting value for the maxima: min() would be the smallest
  // positive number, which is wrong for an object with negative coordinates.
  float y_min = std::numeric_limits<float>::max();
  float y_max = std::numeric_limits<float>::lowest();
  // Get the middle (x) of the object
  float x_sum = 0;
  float x_min = std::numeric_limits<float>::max();
  float x_max = std::numeric_limits<float>::lowest();
  for (const auto& point : object.points) {
    y_min = std::min(y_min, point.y);
    y_max = std::max(y_max, point.y);
    x_sum += point.x;
    x_min = std::min(x_min, point.x);
    x_max = std::max(x_max, point.x);
  }
  const float x_mean = x_sum / static_cast<float>(object.size());
  const float object_height = y_max - y_min;
  const float object_width = x_max - x_min;
  // Y calculated this way for always targeting below halfway
  const float object_height_center = y_min + (object_height / 2);
  const float object_band_start = object_height_center + (object_height / 2.0f * settings.start);
  const float object_band_stop = object_height_center + (object_height / 2.0f * settings.stop);
  const float object_band_width = object_width * settings.width;

  // Get the points that are average or above the middle of the object (because y is flipped we want the bottom part)
  // Also as we are iterating through the points, store the closest one (in the z direction) to the camera
  // Trim any points that are on the sides of the object
  bool found = false;
  float closest_distance = std::numeric_limits<float>::max();
  for (const auto& point : object.points) {
    // only consider the center target area
    if (point.y >= object_band_start && point.y <= object_band_stop
        && point.x >= x_mean - object_band_width && point.x <= x_mean + object_band_width) {
      if (band != nullptr) {
        band->push_back(point);
      }
      if (point.z < closest_distance) {
        closest_distance = point.z;
        closest_point = Eigen::Vector3f(point.x, point.y, point.z);
        found = true;
      }
    }
  }

  if (band != nullptr) {
    band->width = static_cast<std::uint32_t>(band->size());
    band->height = 1;
    band->is_dense = false;
  }
  return found;
}

}  // namespace pcl_pose
