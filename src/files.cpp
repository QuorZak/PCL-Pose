#include "pcl_pose/files.h"

#include <algorithm>
#include <filesystem>

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
