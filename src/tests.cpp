#include <pose_estimation.h>
#include <cstdlib>
#include <thread>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/console/print.h>
#include <opencv2/opencv.hpp>
#include <librealsense2/rs.hpp>

void showPdcFile(const int usePclViewer = 0) {
  const std::string file = "../lab_data/cloud_cluster_0001.pcd";

  if (usePclViewer == 1) {
    const std::string command = "pcl_viewer -multiview 1 " + file;
    if (const int result = std::system(command.c_str()); result != 0) {
      std::cerr << "Failed to execute pcl_viewer command" << std::endl;
    }
    return;
  }

  if (usePclViewer == 2) {
    const std::string command = "pcl_viewer ../data/901.125.07/*.pcd";
    if (const int result = std::system(command.c_str()); result != 0) {
      std::cerr << "Failed to execute pcl_viewer command" << std::endl;
    }
    return;
  }

  pcl::PCLPointCloud2 cloud;
  pcl::console::print_highlight(stderr, "Loading ");
  pcl::console::print_value(stderr, "%s ", file);
  if (pcl::io::loadPCDFile(file, cloud) != 0) {
    std::cerr << "Failed to load " + file << std::endl;
  } else {
    std::cout << "Successfully loaded " + file << std::endl;
  }

  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_xyz(new pcl::PointCloud<pcl::PointXYZ>);
  pcl::fromPCLPointCloud2(cloud, *cloud_xyz);

  pcl::visualization::PCLVisualizer::Ptr viewer(new pcl::visualization::PCLVisualizer("3D Viewer"));
  viewer->setBackgroundColor(0, 0, 0);
  viewer->addPointCloud<pcl::PointXYZ>(cloud_xyz, file);
  viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, file);
  viewer->addCoordinateSystem(1.0);
  viewer->initCameraParameters();

  while (!viewer->wasStopped()) {
    viewer->spinOnce(100);
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
  }
}

void streamDepthMap() {
  rs2::pipeline pipe;
  rs2::config cfg;
  cfg.enable_stream(RS2_STREAM_DEPTH, 640, 480, RS2_FORMAT_Z16, 15);
  pipe.start(cfg);

  while (true) {
    auto frames = pipe.wait_for_frames();
    auto depth = frames.get_depth_frame();

    // Implement a depth threshold filter
    rs2::threshold_filter threshold_filter;
    threshold_filter.set_option(RS2_OPTION_MIN_DISTANCE, depth_filter_min_distance);
    threshold_filter.set_option(RS2_OPTION_MAX_DISTANCE, depth_filter_max_distance);
    depth = threshold_filter.process(depth);

    // Get size of the depth frame
    auto width = depth.get_width();
    auto height = depth.get_height();

    // Use the cropping function from the header
    const auto [x_start, x_stop, y_start, y_stop]
          = get_crop_points(width, height, image_reduced_to_percentage);

    // Create OpenCV matrix of size (w,h) from the depth frame
    // Then crop the image to the specified region
    cv::Mat depth_image(cv::Size(width, height), CV_16UC1, const_cast<void*>(depth.get_data()), cv::Mat::AUTO_STEP);
    cv::Mat cropped_depth_image = depth_image(cv::Rect(x_start, y_start, x_stop - x_start, y_stop - y_start));

    cv::Mat depth_image_8u;
    cropped_depth_image.convertTo(depth_image_8u, CV_8UC1, 255.0 / 10000);

    cv::Mat depth_colormap;
    cv::applyColorMap(depth_image_8u, depth_colormap, cv::COLORMAP_JET);

    cv::imshow("Depth Map", depth_colormap);
    if (cv::waitKey(1) >= 0) {
      break;
    }
  }
  pipe.stop();
  cv::destroyAllWindows();
}

int main() {
  streamDepthMap();
  return 0;
}