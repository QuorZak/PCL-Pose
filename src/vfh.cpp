#include "pcl_pose/vfh.h"

#include <cstring>
#include <sstream>
#include <stack>

#include <pcl/common/io.h>
#include <pcl/console/print.h>
#include <pcl/features/normal_3d.h>
#include <pcl/features/vfh.h>
#include <pcl/io/pcd_io.h>
#include <pcl/search/kdtree.h>

namespace pcl_pose {

void estimate_VFH(const pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud, pcl::PointCloud <pcl::VFHSignature308> &signature)
{
  const pcl::PointCloud<pcl::Normal>::Ptr normals (new pcl::PointCloud<pcl::Normal> ());

  // Where the camera was, in the cloud's own coordinates. Part of the signature (and which way the normals point)
  // depends on it, so it must be the real camera position: the origin for a cloud in the camera frame, or wherever a
  // centred library cloud's camera ended up (stored in sensor_origin_, the VIEWPOINT of the .pcd file).
  const float vp_x = cloud->sensor_origin_[0];
  const float vp_y = cloud->sensor_origin_[1];
  const float vp_z = cloud->sensor_origin_[2];

  // Estimate point cloud normals
  const pcl::search::KdTree<pcl::PointXYZ>::Ptr normalTree (new pcl::search::KdTree<pcl::PointXYZ> ());
  pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> ne;
  ne.setInputCloud (cloud);
  ne.setSearchMethod (normalTree);
  ne.setRadiusSearch (0.03);
  ne.setViewPoint (vp_x, vp_y, vp_z);
  ne.compute(*normals);

  // Estimate VFH signature of the point cloud
  pcl::search::KdTree<pcl::PointXYZ>::Ptr tree (new pcl::search::KdTree<pcl::PointXYZ> ());
  pcl::VFHEstimation<pcl::PointXYZ, pcl::Normal, pcl::VFHSignature308> vfh;
  vfh.setInputCloud (cloud);
  vfh.setInputNormals (normals);
  vfh.setSearchMethod (tree);
  vfh.setViewPoint (vp_x, vp_y, vp_z);
  vfh.compute (signature);
}

bool load_vfh_histogram(const std::filesystem::path &path, vfh_model &vfh)
{
  int vfh_idx;
  // Read file header to check if the file contains VFH signature
  try
  {
    pcl::PCLPointCloud2 cloud;
    int version;
    Eigen::Vector4f origin;
    Eigen::Quaternionf orientation;
    pcl::PCDReader r;
    int type; unsigned int idx;
    r.readHeader (path.string (), cloud, origin, orientation, version, type, idx);

    vfh_idx = pcl::getFieldIndex (cloud, "vfh");
    if (vfh_idx == -1)
    {
      return (false);
    }
    if (static_cast<int>(cloud.width) * cloud.height != 1)
      return (false);
  }
  catch (const pcl::InvalidConversionException&)
  {
    return (false);
  }

  // Treat the VFH signature as a single Point Cloud and load data from it
  pcl::PointCloud <pcl::VFHSignature308> point;
  if (pcl::io::loadPCDFile (path.string (), point) == -1 || point.empty ())
  {
    return (false);
  }
  vfh.second.resize (308);

  // The fields of the VFHSignature308 type (the overload that also took the cloud was removed in PCL 1.12)
  std::vector <pcl::PCLPointField> fields;
  const int field_idx = pcl::getFieldIndex<pcl::VFHSignature308> ("vfh", fields);
  if (field_idx < 0)
  {
    return (false);
  }

  // Fill vfh_model.second with histogram data
  for (size_t i = 0; i < fields[field_idx].count && i < vfh.second.size (); ++i)
  {
    vfh.second[i] = point.points[0].histogram[i];
  }
  // Put file path in vfh_model.first
  vfh.first = path.string ();
  return (true);
}

void load_vfh_model_data(const std::filesystem::path &base_dir, std::vector<vfh_model> &models,
  std::vector<std::string> &model_files) {
  namespace fs = std::filesystem;

  if (!fs::exists(base_dir) || !fs::is_directory(base_dir)) {
    return;
  }

  std::stack<fs::path> directories;
  directories.push(base_dir);

  while (!directories.empty()) {
    const fs::path current_dir = directories.top();
    directories.pop();

    int vfh_count = 0;

    for (const auto& entry : fs::directory_iterator(current_dir)) {
      if (entry.is_directory()) {
        directories.push(entry.path());
      } else if (entry.is_regular_file() && entry.path().extension() == ".pcd") {
        vfh_model m;
        if (load_vfh_histogram(entry.path(), m)) {
          models.push_back(m);
          model_files.push_back(entry.path().string()); // Store the file name
          vfh_count++;
        }
      }
    }

    if (vfh_count > 0) {
      std::stringstream ss;
      ss << "Loaded " << vfh_count << " models from \"" << current_dir.string() << "\"";
      pcl::console::print_highlight("%s\n", ss.str().c_str());
    }
  }
}

}  // namespace pcl_pose
