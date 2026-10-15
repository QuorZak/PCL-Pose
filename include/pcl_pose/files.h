#pragma once
#include <string>
#include <vector>

namespace pcl_pose {

// Returns the full paths of the .pcd files in a folder whose file name starts with the given prefix, sorted by name.
// E.g. listPcdFiles("../lab_data/test/", "spray_bottle_tall_") returns both the clouds and their _vfh files.
std::vector<std::string> listPcdFiles(const std::string& folder, const std::string& prefix = "");

// Returns the next free capture number for an object: one more than the highest NNNN found in files named
// <object_name>_NNNN.pcd or <object_name>_NNNN_vfh.pcd in the folder (1 if there are none). Files that do not
// follow that pattern are ignored, and the folder is created if it does not exist yet.
int nextFileNumber(const std::string& folder, const std::string& object_name);

}  // namespace pcl_pose
