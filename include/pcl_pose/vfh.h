#pragma once
// Parts of these functions were adapted from the PCL tutorial "Cluster Recognition and 6DOF Pose Estimation using
// VFH descriptors" (BSD license, see THIRD_PARTY_NOTICES.md) via dkebude/PCL-object-recognition.
#include <filesystem>
#include <string>
#include <utility>
#include <vector>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

namespace pcl_pose {

// Define vfh_model type for storing data file name and histogram data
typedef std::pair<std::string, std::vector<float> > vfh_model;

// This function estimates VFH signatures of an input point cloud
// Input: The point cloud. Its sensor_origin_ must be where the camera was in the cloud's coordinates: the origin (the
// default) for a cloud in the camera frame. A cloud that has been moved, such as a centred library cloud, must have had
// its sensor_origin_ moved with it (see transformCloudAndViewpoint in library.h), or the signature will not match what
// the camera sees.
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

// To find the library models closest to a signature, use VfhLibrary (vfh_library.h)

}  // namespace pcl_pose
