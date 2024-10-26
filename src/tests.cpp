#include <pose_estimation.h>
#include <cstdlib>
#include <thread>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/console/print.h>
#include <opencv2/opencv.hpp>
#include <librealsense2/rs.hpp>

void showPdcFile(const int usePclViewer = 0) {
  const std::string file = "../lab_data/mustard_small/*.pcd";

  if (usePclViewer == 1) { // Show the file
    const std::string command = "pcl_viewer " + file;
    if (const int result = std::system(command.c_str()); result != 0) {
      std::cerr << "Failed to execute pcl_viewer command" << std::endl;
    }
    return;
  }

  if (usePclViewer == 2) { // Show all files in a split multiview
    const std::string command = "pcl_viewer  -multiview 1 " + file;
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
  // Initialise the Realsense pipeline
  rs2::pipeline pipe;
  rs2::config config;
  config.enable_stream(RS2_STREAM_DEPTH, cam_res_width, cam_res_height, RS2_FORMAT_Z16, cam_fps); // Use global parameters
  pipe.start(config);

  while (true) {
    auto frames = pipe.wait_for_frames();
    auto depth = frames.get_depth_frame();

    // Apply a depth threshold filter
    depth = apply_threshold_filter(depth);

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
    // If the user presses a key
    if (cv::waitKey(1) >= 0) {
      break;
    }
  }
  pipe.stop();
  cv::destroyAllWindows();
}

void stream_point_cloud()
{
  // this function calls the stream_point_cloud_show_depth_map function just to test the video stream
  // Do nothing with the point cloud
  rs2::pipeline pipe;
  pcl::PointCloud<pcl::PointXYZ>::Ptr output_stream_cloud(new pcl::PointCloud<pcl::PointXYZ>);
  std::mutex mtx;
  std::condition_variable condition_var;
  bool ready = false;

  std::thread img_thread(stream_point_cloud_show_depth_map, std::ref(pipe), std::ref(output_stream_cloud),
      std::ref(mtx), std::ref(condition_var), std::ref(ready));

  {
    std::unique_lock<std::mutex> lock(mtx);
    condition_var.wait(lock, [&ready] { return ready; });
  }
  // wait for the thread to finish
  img_thread.join();
}

int main() {
  //showPdcFile(1);

  //streamDepthMap();

  stream_point_cloud();
  return 0;
}