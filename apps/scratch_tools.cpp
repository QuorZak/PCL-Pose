#include <pose_estimation.h>

using namespace pcl_pose;
#include <cstdlib>
#include <thread>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/console/print.h>


void showPdcFile(const int usePclViewer = 0) {
  //const std::string file = "../lab_data/test/spray_bottle_tall_0008.pcd";
  const std::string file = "../lab_data/spray_bottle_tall/spray_bottle_tall_0007.pcd";

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
  pcl::console::print_value(stderr, "%s ", file.c_str());
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
    depth = apply_depth_post_processing_filters(depth);

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

void find_best_match()
{
  // Load the VFH model library and build its search index
  // (before starting the camera, so there is nothing to shut down if the library is empty)
  const VfhLibrary library(model_directory);
  if (library.empty()) {
    std::cerr << "No VFH models found in " << model_directory << std::endl;
    return;
  }

  // Initialize the RealSense pipeline
  rs2::pipeline pipe;
  rs2::config config;
  config.enable_stream(RS2_STREAM_DEPTH, cam_res_width, cam_res_height, RS2_FORMAT_Z16, cam_fps); // Use global parameters
  pipe.start(config);

  // Initialize the filters which will be applied to the depth frame
  initialise_filters();

  // Camera warmup - dropping several first frames to let auto-exposure stabilize
  for (int i = 0; i < 100; i++) {
    auto frames = pipe.wait_for_frames(); // purposely does nothing with the frames
  }
  std::cout << "Ready to capture point cloud." << std::endl;

  // Capture a frame
  rs2::frameset frames = pipe.wait_for_frames();
  rs2::frame depth = frames.get_depth_frame();
  rs2::frame filtered_depth = apply_depth_post_processing_filters(depth);

  // Convert the depth frame to a PCL point cloud
  pcl::PointCloud<pcl::PointXYZ>::Ptr captured_cloud = depthFrameToPointCloud(filtered_depth, true, true);

  // Segment the point cloud
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_filtered(new pcl::PointCloud<pcl::PointXYZ>);
  std::vector<pcl::PointIndices> cluster_indices;
  filterAndSegmentPointCloud(captured_cloud, cloud_filtered, cluster_indices, true);

  // Compare the captured model to the models
  pcl::PointCloud<pcl::PointXYZ>::Ptr best_cluster(new pcl::PointCloud<pcl::PointXYZ>);
  float best_distance = std::numeric_limits<float>::max();
  std::string best_model_file; // Stays empty until a cluster matches a model

  for (const auto& cluster : cluster_indices) {
    int k = 6;
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_cluster(new pcl::PointCloud<pcl::PointXYZ>);
    for (const auto& idx : cluster.indices) {
      cloud_cluster->push_back(cloud_filtered->points[idx]);
    }

    // Translate the point cloud to the origin (centroid), so it shows on top of the (centred) library cloud. The
    // viewpoint moves with it, so the VFH signature is still the one the camera sees.
    centreOnCentroid(*cloud_cluster);

    pcl::PointCloud<pcl::VFHSignature308> signature;
    estimate_VFH(cloud_cluster, signature);

    vfh_model query_model;
    query_model.second.assign(signature.points[0].histogram, signature.points[0].histogram + 308);
    const std::vector<VfhLibrary::Match> matches = library.nearest(query_model, k);

    if (!matches.empty() && matches[0].distance < best_distance) {
      best_distance = matches[0].distance;
      best_model_file = matches[0].file;
      *best_cluster = *cloud_cluster;
    }
  }

  if (!best_model_file.empty()) {
    // Load the best match model
    pcl::PointCloud<pcl::PointXYZ>::Ptr best_match_cloud(new pcl::PointCloud<pcl::PointXYZ>);
    const std::string best_match_name = cloudFileForVfh(best_model_file);
    pcl::io::loadPCDFile(best_match_name, *best_match_cloud);

    std::cout << "Best match name: " << best_match_name << std::endl;

    // Visualize both point clouds
    pcl::visualization::PCLVisualizer::Ptr viewer(new pcl::visualization::PCLVisualizer("3D Viewer"));
    viewer->setBackgroundColor(0, 0, 0);
    viewer->addPointCloud<pcl::PointXYZ>(best_cluster, "best_cluster");
    viewer->addPointCloud<pcl::PointXYZ>(best_match_cloud, "best_match_cloud");
    viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 5, "best_cluster");
    viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 5, "best_match_cloud");
    viewer->initCameraParameters();

    while (!viewer->wasStopped()) {
      viewer->spinOnce(100);
      std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
  }

  pipe.stop();
}

int main() {
  //showPdcFile(2);

  //streamDepthMap();

  // 1258020693333_cluster_1_nxyz.pcd
  // 1258020693333_cluster_0_nxyz.pcd
  //const auto files = listPcdFiles("../lab_data/test/");
  showPointClouds({"../lab_data/test/spray_bottle_tall_0010.pcd"}, true);
  //find_best_match();

  return 0;
}
