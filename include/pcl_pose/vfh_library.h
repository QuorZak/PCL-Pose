#pragma once
#include <cstddef>
#include <memory>
#include <string>
#include <vector>

#include <flann/flann.h>

#include "pcl_pose/vfh.h"

namespace pcl_pose {

// The VFH signatures of an object library, with an index for finding the ones closest to a query signature.
// It owns all its memory.
class VfhLibrary {
public:
  struct Match {
    std::size_t index; // Position of the model in modelFiles()
    float distance;    // Chi-square distance to the query, 0 is identical
    std::string file;  // The _vfh.pcd file of the model (see cloudFileForVfh for its point cloud)
  };

  // Loads every _vfh.pcd in the folder and its sub-folders, and builds the index. Check empty() afterwards.
  explicit VfhLibrary(const std::string& folder);

  VfhLibrary(const VfhLibrary&) = delete;
  VfhLibrary& operator=(const VfhLibrary&) = delete;

  [[nodiscard]] bool empty() const { return models_.empty(); }
  [[nodiscard]] std::size_t size() const { return models_.size(); }
  [[nodiscard]] const std::vector<std::string>& modelFiles() const { return model_files_; }

  // The k models closest to the query, best first. Fewer are returned if the library is smaller than k, and none if
  // the library is empty or the query does not have the same number of bins as the models.
  [[nodiscard]] std::vector<Match> nearest(const vfh_model& query, int k) const;

private:
  std::vector<vfh_model> models_;
  std::vector<std::string> model_files_;
  std::vector<float> data_; // All the signatures in one block: the index points at this memory, so it must not move
  std::unique_ptr<flann::Index<flann::ChiSquareDistance<float>>> index_;
};

}  // namespace pcl_pose
