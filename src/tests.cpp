#include <pose_estimation.h>
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
  // Models
  std::vector<vfh_model> models;

  // Other required variables
  Matrix<int> k_indices;
  Matrix<float> k_distances;

  // Initialize the RealSense pipeline
  rs2::pipeline pipe;
  rs2::config config;
  config.enable_stream(RS2_STREAM_DEPTH, cam_res_width, cam_res_height, RS2_FORMAT_Z16, cam_fps); // Use global parameters
  pipe.start(config);

  // Initialize the filters which will be applied to the depth frame
  initialise_filters();

  // Load VFH model data
  std::vector<std::string> model_files;
  load_vfh_model_data(model_directory, models, model_files);

  // Convert models to FLANN format
  std::unique_ptr<float[]> data_ptr(new float[models.size() * models[0].second.size()]);
  Matrix data(data_ptr.get(), models.size(), models[0].second.size());
  for (size_t i = 0; i < data.rows; ++i)
    for (size_t j = 0; j < data.cols; ++j)
      data[i][j] = models[i].second[j];

  // Build the FLANN index
  Index<ChiSquareDistance<float>> index(data, LinearIndexParams());
  index.buildIndex();

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
  int best_index = -1;

  for (const auto& cluster : cluster_indices) {
    int k = 6;
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_cluster(new pcl::PointCloud<pcl::PointXYZ>);
    for (const auto& idx : cluster.indices) {
      cloud_cluster->push_back(cloud_filtered->points[idx]);
    }

    // Compute the centroid of the point cloud
    Eigen::Vector4f centroid;
    compute3DCentroid(*cloud_cluster, centroid);

    // Translate the point cloud to the origin (centroid)
    Eigen::Affine3f transform = Eigen::Affine3f::Identity();
    transform.translation() << -centroid[0], -centroid[1], -centroid[2];
    transformPointCloud(*cloud_cluster, *cloud_cluster, transform);

    pcl::PointCloud<pcl::VFHSignature308> signature;
    estimate_VFH(cloud_cluster, signature);

    vfh_model query_model;
    query_model.second.assign(signature.points[0].histogram, signature.points[0].histogram + 308);
    nearestKSearch(index, query_model, k, k_indices, k_distances);

    if (k_distances[0][0] < best_distance) {
      best_distance = k_distances[0][0];
      best_index = k_indices[0][0];
      *best_cluster = *cloud_cluster;
    }
  }

  if (best_index != -1) {
    // Load the best match model
    pcl::PointCloud<pcl::PointXYZ>::Ptr best_match_cloud(new pcl::PointCloud<pcl::PointXYZ>);
    std::string best_match_name = models[best_index].first.substr(0, models[best_index].first.find("_vfh.pcd")) + ".pcd";
    pcl::io::loadPCDFile(best_match_name, *best_match_cloud);

    cout << "Best match name: " << best_match_name << std::endl;

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

/*// Move the arm forward by x cm
void moveArmForward(float x_change) {
  ros::NodeHandle node_handle;
  ros::AsyncSpinner spinner(1);
  spinner.start();

  moveit::planning_interface::MoveGroupInterface move_group("manipulator");
  moveit::planning_interface::PlanningSceneInterface planning_scene_interface;

  geometry_msgs::PoseStamped current_pose = move_group.getCurrentPose();
  geometry_msgs::Pose target_pose = current_pose.pose;
  target_pose.position.x += x_change; // Move forward by x cm

  move_group.setPoseTarget(target_pose);

  moveit::planning_interface::MoveGroupInterface::Plan my_plan;
  bool success = (move_group.plan(my_plan) == moveit::planning_interface::MoveItErrorCode::SUCCESS);

  if (success) {
    move_group.move();
    ROS_INFO("Move successful");
  } else {
    ROS_WARN("Move failed");
  }

  ros::shutdown();
}*/

int main() {
  //showPdcFile(2);

  streamDepthMap();

  // 1258020693333_cluster_1_nxyz.pcd
  // 1258020693333_cluster_0_nxyz.pcd
  //auto files = globFiles("/home/zak/Repos/Zak_POSE/lab_data/spray_bottle_tall/spray_bottle_tall_0007.pcd");
  //showPointClouds(files, true);
  //find_best_match();

  /*ros::init(argc, argv, "move_arm_forward_test");
  moveArmForward(0.02f);*/

  return 0;
}