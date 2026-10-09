#include "pcl_pose/viewer.h"

#include <chrono>
#include <iostream>
#include <thread>

#include <pcl/console/print.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/visualization/pcl_visualizer.h>

void showPointClouds(const std::vector<std::string>& created_files, const bool ignore_vfh) {
  const pcl::visualization::PCLVisualizer::Ptr viewer(new pcl::visualization::PCLVisualizer("3D Viewer"));
  viewer->setBackgroundColor(0, 0, 0);

  for (const auto& file : created_files) {
    std::string pcd_file = file;
    if (file.substr(file.find_last_of(".") + 1) == "pcd" && file.find("_vfh.pcd") != std::string::npos) {
      if (ignore_vfh) {
        continue;
      }
      // If the file is a VFH file, then we need to load the original file
      pcd_file = file.substr(0, file.find("_vfh.pcd")) + ".pcd";
    }
    std::cout << "Loading .pcd file for visualisation: " << pcd_file << std::endl;
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
    if (pcl::io::loadPCDFile(pcd_file, *cloud) == -1) {
      PCL_ERROR("Couldn't read file %s \n", pcd_file.c_str());
      continue;
    }
    viewer->addPointCloud<pcl::PointXYZ>(cloud, pcd_file);
    viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 5, pcd_file); // Increase point size
  }

  // Add coordinate axes to the viewer
  viewer->setBackgroundColor(0, 0, 0);
  viewer->addCoordinateSystem(0.1);
  viewer->initCameraParameters();

  while (!viewer->wasStopped()) {
    viewer->spinOnce(100);
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }
}
