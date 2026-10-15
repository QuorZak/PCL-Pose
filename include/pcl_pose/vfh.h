#pragma once
// Parts of these functions were adapted from the PCL tutorial "Cluster Recognition and 6DOF Pose Estimation using
// VFH descriptors" (BSD license, see THIRD_PARTY_NOTICES.md) via dkebude/PCL-object-recognition.
#include <filesystem>
#include <string>
#include <utility>
#include <vector>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#include <flann/flann.h>
#include <flann/io/hdf5.h>

namespace pcl_pose {

// Define vfh_model type for storing data file name and histogram data
typedef std::pair<std::string, std::vector<float> > vfh_model;

// This function estimates VFH signatures of an input point cloud
// Input: The point cloud
// Output: VFH signature of the object
void estimate_VFH(const pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud, pcl::PointCloud <pcl::VFHSignature308> &signature);

// This function loads VFH signature histogram into a vfh model
// Input: File path to histogram
// Output: A boolean that returns true if the histogram is loaded successfully and a vfh_model data that holds the histogram information
bool load_vfh_histogram(const std::filesystem::path &path, vfh_model &vfh);

// This gets PCD file names from a directory (and its sub-directories) and passes them to histogram loader one by one
// then pushes the results into a vfh_model vector to keep them in a data storage
// Input: Model data set file path
// Output: A vfh_model vector that contains all VFH signature information, and the file name of each model
void load_vfh_model_data(const std::filesystem::path &base_dir, std::vector<vfh_model> &models,
  std::vector<std::string> &model_files);

// This function finds the k nearest neighbors of the query object and returns their indices in the K-d tree and distances from the query object
// Inputs: K-d tree, query object model, desired candidate count (k)
// Outputs: K-d tree indices of candidates, Distances of candidates from the query object
// The two result matrices are allocated by this function: start with default-constructed matrices, and reuse the same
// ones for every call (each call frees the previous allocation). Do not pass in matrices that point at memory
// owned elsewhere.
void nearestKSearch(const flann::Index<flann::ChiSquareDistance<float> > &index, const vfh_model &model,
  int k, flann::Matrix<int> &indices, flann::Matrix<float> &distances);

}  // namespace pcl_pose
