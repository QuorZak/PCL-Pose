#pragma once
#include <string>
#include <vector>

namespace pcl_pose {

// Function to show the point clouds with increased point size
// A _vfh.pcd file is shown as the cloud it belongs to, unless ignore_vfh is set, in which case it is skipped
void showPointClouds(const std::vector<std::string>& created_files, bool ignore_vfh = false);

}  // namespace pcl_pose
