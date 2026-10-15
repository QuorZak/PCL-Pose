#include "pcl_pose/files.h"

#include <algorithm>
#include <cctype>
#include <filesystem>

namespace pcl_pose {

std::vector<std::string> listPcdFiles(const std::string& folder, const std::string& prefix) {
  namespace fs = std::filesystem;
  std::vector<std::string> files;
  if (!fs::is_directory(folder)) {
    return files;
  }
  for (const auto& entry : fs::directory_iterator(folder)) {
    if (!entry.is_regular_file() || entry.path().extension() != ".pcd") {
      continue;
    }
    if (entry.path().filename().string().rfind(prefix, 0) == 0) {
      files.push_back(entry.path().string());
    }
  }
  std::sort(files.begin(), files.end());
  return files;
}

int nextFileNumber(const std::string& folder, const std::string& object_name) {
  namespace fs = std::filesystem;
  fs::create_directories(folder);

  const std::string prefix = object_name + "_";
  int highest = 0;
  for (const auto& file : listPcdFiles(folder, prefix)) {
    // The four digits straight after the prefix are the capture number
    const std::string name = fs::path(file).filename().string();
    const std::string digits = name.substr(prefix.size(), 4);
    if (digits.size() == 4 && std::all_of(digits.begin(), digits.end(), [](unsigned char c) { return std::isdigit(c); })) {
      highest = std::max(highest, std::stoi(digits));
    }
  }
  return highest + 1;
}

}  // namespace pcl_pose
