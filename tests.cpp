// When this file is compiled, it will produce an executable called tests
// that can be run from the command line or the debugger.
// It will contain some basic functions to test the functionality of the
// code in pose.h and pose.cpp.

#include <pose.h>
#include <cstdlib>
#include <thread>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/console/print.h>

void showPCDfile(const int usePclViewer = 0) { // The file saved from main is "capture.pcd"
  if (usePclViewer == 1) {
    const std::string command = "pcl_viewer -multiview 1 capture.pcd";
    if (const int result = std::system(command.c_str()); result != 0) {
      std::cerr << "Failed to execute pcl_viewer command" << std::endl;
    }
    return;
  }

  pcl::PCLPointCloud2 cloud;
  pcl::console::print_highlight(stderr, "Loading ");
  pcl::console::print_value(stderr, "%s ", "capture.pcd");
  if (pcl::io::loadPCDFile("capture.pcd", cloud) != 0) {
    std::cerr << "Failed to load capture.pcd" << std::endl;
  } else {
    std::cout << "Successfully loaded capture.pcd" << std::endl;
  }

  // Convert PCLPointCloud2 to PointCloud<PointXYZ>
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_xyz(new pcl::PointCloud<pcl::PointXYZ>);
  pcl::fromPCLPointCloud2(cloud, *cloud_xyz);

  // Visualize the point cloud
  pcl::visualization::PCLVisualizer::Ptr viewer(new pcl::visualization::PCLVisualizer("3D Viewer"));
  viewer->setBackgroundColor(0, 0, 0);
  viewer->addPointCloud<pcl::PointXYZ>(cloud_xyz, "sample cloud");
  viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "sample cloud");
  viewer->addCoordinateSystem(1.0);
  viewer->initCameraParameters();

  while (!viewer->wasStopped()) {
    viewer->spinOnce(100);
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
  }
}

int main() {
  // Uncomment the function calls to run them
  showPCDfile(1);
  return 0;
}