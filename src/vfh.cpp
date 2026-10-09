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

void estimate_VFH(const pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud, pcl::PointCloud <pcl::VFHSignature308> &signature)
{
  const pcl::PointCloud<pcl::Normal>::Ptr normals (new pcl::PointCloud<pcl::Normal> ());

  // Estimate point cloud normals
  const pcl::search::KdTree<pcl::PointXYZ>::Ptr normalTree (new pcl::search::KdTree<pcl::PointXYZ> ());
  pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> ne;
  ne.setInputCloud (cloud);
  ne.setSearchMethod (normalTree);
  ne.setRadiusSearch (0.03);
  ne.compute(*normals);

  // Estimate VFH signature of the point cloud
  pcl::search::KdTree<pcl::PointXYZ>::Ptr tree (new pcl::search::KdTree<pcl::PointXYZ> ());
  pcl::VFHEstimation<pcl::PointXYZ, pcl::Normal, pcl::VFHSignature308> vfh;
  vfh.setInputCloud (cloud);
  vfh.setInputNormals (normals);
  vfh.setSearchMethod (tree);
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
  pcl::io::loadPCDFile (path.string (), point);
  vfh.second.resize (308);

  std::vector <pcl::PCLPointField> fields;
  pcl::getFieldIndex (point, "vfh", fields);

  // Fill vfh_model.second with histogram data
  for (size_t i = 0; i < fields[vfh_idx].count; ++i)
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

void nearestKSearch(const flann::Index<flann::ChiSquareDistance<float> > &index, const vfh_model &model,
  const int k, flann::Matrix<int> &indices, flann::Matrix<float> &distances)
{
  // Query point
  flann::Matrix<float> p = flann::Matrix<float>(new float[model.second.size ()], 1, model.second.size ());
  memcpy (&p.ptr ()[0], &model.second[0], p.cols * p.rows * sizeof (float));

  indices = flann::Matrix<int>(new int[k], 1, k);
  distances = flann::Matrix<float>(new float[k], 1, k);
  index.knnSearch (p, indices, distances, k, flann::SearchParams (512));
  delete[] p.ptr ();
}
