#include <pose_estimation.h>
#include <cstdlib>
#include <thread>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/console/print.h>

#include <opencv2/objdetect/aruco_detector.hpp>

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
    depth = apply_post_processing_filters(depth);

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
  load_vfh_model_data(model_directory, models);

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
  rs2::frame filtered_depth = apply_post_processing_filters(depth);

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

// Generate and save openCv aruco markers
void generateMarker() {
  const cv::aruco::Dictionary dictionary = cv::aruco::getPredefinedDictionary(aruco_dict_number);
  cv::Mat markerImage;
  // Start with aruco_marker_id, then do it 7 more times (8 total), add 11 each time
  for (int marker_id = aruco_marker_id; marker_id <= 88; marker_id = marker_id + 11) {
    generateImageMarker(dictionary, marker_id, aruco_marker_pixels, markerImage, 1);
    std::string f_marker_name = "marker_" + std::to_string(marker_id)
      + "_dict_" + std::to_string(aruco_dict_number) + "_size_" + std::to_string(aruco_marker_pixels) + ".png";
    const std::string filename = f_markers_location + f_marker_name;
    imwrite(filename, markerImage);
  }
}

// function to start up camera rgb, find specifically the marker generated in generate marker, and display it
void findMarkerAndPose() {
using namespace cv;
// Initialize the RealSense pipeline
rs2::pipeline pipe;
rs2::config config;
config.enable_stream(RS2_STREAM_COLOR, cam_res_width, cam_res_height, RS2_FORMAT_BGR8, cam_fps); // Use global parameters
pipe.start(config);

// Get the intrinsics of the color stream
rs2::pipeline_profile profile = pipe.get_active_profile();
auto color_stream = profile.get_stream(RS2_STREAM_COLOR).as<rs2::video_stream_profile>();
rs2_intrinsics intrinsics = color_stream.get_intrinsics();

// Camera matrix and distortion coefficients
cv::Mat camMatrix = (cv::Mat_<double>(3, 3) << intrinsics.fx, 0, intrinsics.ppx, 0, intrinsics.fy, intrinsics.ppy, 0, 0, 1);
cv::Mat distCoeffs = (cv::Mat_<double>(5, 1) << intrinsics.coeffs[0], intrinsics.coeffs[1], intrinsics.coeffs[2], intrinsics.coeffs[3], intrinsics.coeffs[4]);

  // set coordinate system
  float markerLength = 0.1f;
  cv::Mat objPoints(4, 1, CV_32FC3);
  objPoints.ptr<Vec3f>(0)[0] = Vec3f(-markerLength/2.f, markerLength/2.f, 0);
  objPoints.ptr<Vec3f>(0)[1] = Vec3f(markerLength/2.f, markerLength/2.f, 0);
  objPoints.ptr<Vec3f>(0)[2] = Vec3f(markerLength/2.f, -markerLength/2.f, 0);
  objPoints.ptr<Vec3f>(0)[3] = Vec3f(-markerLength/2.f, -markerLength/2.f, 0);

  // Create the detector
  cv::aruco::DetectorParameters detectorParams = cv::aruco::DetectorParameters();
  cv::aruco::Dictionary dictionary = aruco::getPredefinedDictionary(aruco_dict_number);
  cv::aruco::ArucoDetector detector(dictionary, detectorParams);

while (true) {
  // Get the frames
  auto frames = pipe.wait_for_frames();
  auto color_frame = frames.get_color_frame();

  // Convert the frame to an OpenCV image
  cv::Mat frame(cv::Size(cam_res_width, cam_res_height), CV_8UC3, const_cast<void*>(color_frame.get_data()), cv::Mat::AUTO_STEP);

  // Detect the marker
  std::vector<int> markerIds;
  std::vector<std::vector<cv::Point2f>> markerCorners, rejectedCandidates;
  detector.detectMarkers(frame, markerCorners, markerIds, rejectedCandidates);

  size_t nMarkers = markerCorners.size();
  std::vector<Vec3d> rvecs(nMarkers), tvecs(nMarkers);

  if(estimatePose && !markerIds.empty()) {
    // Calculate pose for each marker
    for (size_t i = 0; i < nMarkers; i++) {
      solvePnP(objPoints, markerCorners.at(i), camMatrix, distCoeffs, rvecs.at(i), tvecs.at(i));
    }
  }

  // If the marker is detected, draw the marker
  if (!markerIds.empty()) {
    cv::aruco::drawDetectedMarkers(frame, markerCorners, markerIds);
    for(unsigned int i = 0; i < markerIds.size(); i++)
      cv::drawFrameAxes(frame, camMatrix, distCoeffs, rvecs[i], tvecs[i], markerLength * 1.5f, 2);
  }

  // Display the frame
  cv::imshow("Frame", frame);

  // If the user presses a key
  if (cv::waitKey(1) >= 0) {
    break;
  }
}
pipe.stop();
destroyAllWindows();
}







int main() {
  //showPdcFile(1);

  //streamDepthMap();

  // 1258020693333_cluster_1_nxyz.pcd
  // 1258020693333_cluster_0_nxyz.pcd
  //auto files = globFiles("/home/zak/Repos/Zak_POSE/lab_data/spray_bottle_tall/spray_bottle_tall_0003.pcd");
  //showPointClouds(files, true);
  //find_best_match();

  //generateMarker();
  findMarkerAndPose();


  return 0;
}