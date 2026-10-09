#pragma once
#include <string>
#include <vector>

// Returns the full paths of the .pcd files in a folder whose file name starts with the given prefix, sorted by name.
// E.g. listPcdFiles("../lab_data/test/", "spray_bottle_tall_") returns both the clouds and their _vfh files.
std::vector<std::string> listPcdFiles(const std::string& folder, const std::string& prefix = "");
