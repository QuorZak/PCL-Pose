#include "pcl_pose/vfh_library.h"

#include <algorithm>

namespace pcl_pose {

VfhLibrary::VfhLibrary(const std::string& folder) {
  load_vfh_model_data(folder, models_, model_files_);
  if (models_.empty()) {
    return;
  }

  // Put all the signatures in one block of memory for FLANN
  const std::size_t rows = models_.size();
  const std::size_t cols = models_[0].second.size();
  data_.resize(rows * cols);
  for (std::size_t i = 0; i < rows; ++i) {
    for (std::size_t j = 0; j < cols; ++j) {
      data_[i * cols + j] = models_[i].second[j];
    }
  }

  // Build the FLANN index. This is the data structure that is used to search for the best match
  const flann::Matrix<float> dataset(data_.data(), rows, cols);
  index_ = std::make_unique<flann::Index<flann::ChiSquareDistance<float>>>(dataset, flann::LinearIndexParams());
  index_->buildIndex();
}

std::vector<VfhLibrary::Match> VfhLibrary::nearest(const vfh_model& query, const int k) const {
  std::vector<Match> matches;
  if (index_ == nullptr || k <= 0 || query.second.size() != models_[0].second.size()) {
    return matches;
  }

  const int count = std::min(k, static_cast<int>(models_.size()));
  std::vector<float> query_data(query.second);
  flann::Matrix<float> query_matrix(query_data.data(), 1, query_data.size());
  std::vector<int> indices(count);
  std::vector<float> distances(count);
  flann::Matrix<int> indices_matrix(indices.data(), 1, count);
  flann::Matrix<float> distances_matrix(distances.data(), 1, count);
  index_->knnSearch(query_matrix, indices_matrix, distances_matrix, count, flann::SearchParams(512));

  for (int i = 0; i < count; ++i) {
    const auto index = static_cast<std::size_t>(indices[i]);
    matches.push_back({index, distances[i], model_files_[index]});
  }
  return matches;
}

}  // namespace pcl_pose
