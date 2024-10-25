// When this file is compiled, it will produce an executable called tests
// that can be run from the command line or the debugger.
// It will contain some basic functions to test the functionality of the
// code in pose.h and pose.cpp.

#include <pose_estimation.h>
#include <cstdlib>
#include <thread>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/console/print.h>

void showPdcFile(const int usePclViewer = 0) { // The file saved from main is "../output/capture_vfh.pcd"
  const std::string file = "../lab_data/cloud_cluster_0001.pcd";

  if (usePclViewer == 1) {
    const std::string command = "pcl_viewer -multiview 1 "+ file;
    if (const int result = std::system(command.c_str()); result != 0) {
      std::cerr << "Failed to execute pcl_viewer command" << std::endl;
    }
    return;
  }

  // if usePclViewer == 2, then run a command like this ./pcl_viewer cloud_cluster_0000.pcd cloud_cluster_0001.pcd cloud_cluster_0002.pcd cloud_cluster_0003.pcd cloud_cluster_0004.pcd
  // Set up the function to open all the files in the directory that follow the naming convention cloud_cluster_0000.pcd, cloud_cluster_0001.pcd, etc.
  if (usePclViewer == 2) {
    const std::string command = "pcl_viewer ../lab_data/cloud_cluster_*.pcd";
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

  // Convert PCLPointCloud2 to PointCloud<PointXYZ>
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_xyz(new pcl::PointCloud<pcl::PointXYZ>);
  pcl::fromPCLPointCloud2(cloud, *cloud_xyz);

  // Visualize the point cloud
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
  // We will want to crop the image to focus only on the center of the image (50% in x and y)
  // And also slightly lower in the y direction (10%)
  constexpr int overall_reduction_percentage = 50;
  constexpr int y_downshift_percentage = 10;
  constexpr int x_offset = 640 * overall_reduction_percentage / 100 / 2;
  constexpr int y_offset = 480 * overall_reduction_percentage / 100 / 2 + 480 * y_downshift_percentage / 100;
  constexpr int crop_width = 640 * overall_reduction_percentage / 100;
  constexpr int crop_height = 480 * overall_reduction_percentage / 100;

  // Create a pipeline
  rs2::pipeline pipe;

  // Create a configuration for configuring the pipeline with a non default profile
  rs2::config cfg;

  // Add a stream with its parameters
  cfg.enable_stream(RS2_STREAM_DEPTH, 640, 480, RS2_FORMAT_Z16, 15);

  // Instruct pipeline to start streaming with the requested configuration
  pipe.start(cfg);

  while (true) {
    // Wait for the next set of frames from the camera
    auto frames = pipe.wait_for_frames();

    // Get a frame from the depth stream
    auto depth = frames.get_depth_frame();
    // filter the depth frame
    rs2::threshold_filter threshold_filter;
    threshold_filter.set_option(RS2_OPTION_MIN_DISTANCE, 0.3f);
    threshold_filter.set_option(RS2_OPTION_MAX_DISTANCE, 1.0f);
    depth = threshold_filter.process(depth);

    // Query the distance from the camera to the object in the center of the image
    const float distance = depth.get_distance(320, 240);
    std::cout << "The camera is facing an object " << distance << " meters away \r";

    // Create OpenCV matrix of size (w,h) from the depth frame
    cv::Mat depth_image(cv::Size(640, 480), CV_16UC1, const_cast<void*>(depth.get_data()), cv::Mat::AUTO_STEP);
    cv::Mat cropped_depth_image = depth_image(cv::Rect(x_offset, y_offset, crop_width, crop_height));

    // Convert the depth image to CV_8UC1
    cv::Mat depth_image_8u;
    cropped_depth_image.convertTo(depth_image_8u, CV_8UC1, 255.0 / 10000); // Scale the depth values to 8-bit

    // Apply colormap on depth image
    cv::Mat depth_colormap;
    cv::applyColorMap(depth_image_8u, depth_colormap, cv::COLORMAP_JET);

    // Display the depth map
    cv::imshow("Depth Map", depth_colormap);
    if (cv::waitKey(1) >= 0) {
      break;
    }
  }
  // Stop the pipeline
  pipe.stop();
  cv::destroyAllWindows();
}


int main() {
  // Uncomment the function calls to run them

  showPdcFile(2);

  // Just stream the depth map heat map type image from Realsense through OpenCV
  // streamDepthMap();

  return 0;
}