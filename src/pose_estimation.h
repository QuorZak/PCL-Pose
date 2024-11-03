#pragma once
#include <librealsense2/rs.hpp>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/common/common.h>
#include <pcl/common/transforms.h>
#include <pcl/console/parse.h>
#include <pcl/console/print.h>
#include <pcl/io/pcd_io.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/features/vfh.h>
#include <pcl/features/normal_3d.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/registration/icp.h>

#include <boost/filesystem.hpp>
#include <flann/flann.h>
#include <flann/io/hdf5.h>
#include <iostream>
#include <thread>
#include <mutex>
#include <utility>
#include <condition_variable>
#include <stack>
#include <glob.h>
#include <Eigen/Dense>
#include <cmath>
#include <chrono>

#include <opencv2/opencv.hpp>
#include <opencv2/core/eigen.hpp>
#include <opencv2/imgproc/imgproc.hpp>

/*#include <ros/ros.h>
#include <moveit/move_group_interface/move_group_interface.h>
#include <moveit/planning_scene_interface/planning_scene_interface.h>
#include <moveit_msgs/DisplayRobotState.h>
#include <moveit_msgs/DisplayTrajectory.h>
#include <moveit_msgs/AttachedCollisionObject.h>
#include <moveit_msgs/CollisionObject.h>
#include <geometry_msgs/PoseStamped.h>*/

// Define vfh_model type for storing data file name and histogram data
typedef std::pair<std::string, std::vector<float> > vfh_model;

// Declare global variables that all functions in the .cpp files can access
inline extern const int image_reduced_to_percentage = 70;
inline extern const float depth_filter_min_distance = 0.3f;
inline extern const float depth_filter_max_distance = 0.6f;

// Global camera config params
// 848x480 resolution, 15 frames per second is optimal for Realsense D455
inline extern const int cam_res_width = 848; // Standard 640
inline extern const int cam_res_height = 480; // Standard 480
inline extern const int cam_fps = 15;

// Parameters for cloud filtering
inline extern const float leaf_size = 0.01f; // 0.01f default
inline extern const float cluster_tolerance = 0.02f; // 0.02 default
inline extern const int min_cluster_size = 150; // 100 default
inline extern const int max_cluster_size = 800; // 25000 default
inline extern const float segment_distance_threshold = 0.02f; // 0.02 default
inline extern const float segment_probability = 0.99f; // try 0.99 ? Increase the probability to get a good sample
inline extern const float segment_radius_min = 0.01f; // try 0.01 ? Set radius limits to avoid collinear points
inline extern const float segment_radius_max = 0.1f; // try 0.1 ?
inline extern const float filter_cluster_to_x_percent = 0.99f; // 0.4 default

// Parameters for Pose Estimation
inline extern int icp_max_iterations = 50; // Maximum number of ICP iterations
inline extern float icp_transformation_epsilon = 1e-8; // Transformation epsilon for ICP convergence
inline extern float icp_euclidean_fitness_epsilon = 1; // Euclidean fitness epsilon for ICP convergence

// Parameters for Pose Manager
inline extern float start_scale_distance = 0.01f; // Start distance for scaling in m
inline extern float stop_scale_distance = 0.5f; // Stop distance for scaling in m
enum class PoseUpdateStabilityFactor {
  VeryWilling = 1,
  Willing = 2,
  Linear = 3,
  Resistant = 4,
  VeryResistant = 5
};
inline extern PoseUpdateStabilityFactor stability_factor = PoseUpdateStabilityFactor::Resistant;

// Define the transform to convert from camera frame to robot end effector frame (TCP)
// For now is manually set. The camera is -50mm y, +10mm z, 0 x from the robot frame
// The rotation is the same as the robot frame
inline extern const Eigen::Matrix4f camera_to_TCP = (Eigen::Matrix4f() <<
  1, 0, 0, 0,
  0, 1, 0, -0.05,
  0, 0, 1, 0.01,
  0, 0, 0, 1).finished();

// The current position of the TCP from the base plate (world reference) is:
// x = 305.78, y = 338.90, z = 73.94 mm
// Rotational Vector (Rodrigues), Rotation x = 3.613 rad, Rotation y = -1.863 rad, Rotation z = 0.986 rad
// Store this as an rvec and tvec
inline extern const Eigen::Vector3f base_to_TCP_rvec = Eigen::Vector3f(3.613, -1.863, 0.986);
inline extern const Eigen::Vector3f base_to_TCP_tvec = Eigen::Vector3f(0.30578, 0.3389, 0.07394);

// Camera rotation for home position. This is the rotation that needs to be applied each time we move the camera from home
// The rotation needing to be applied around the x-axis is +22 degrees wrt the tool frame.
inline extern float camera_rotation_home_position = 24.0f;

// Directory for reference models
inline extern std::string model_directory = "../lab_data/spray_bottle_tall/"; // "../lab_data/";

// Parameters for fiducial markers
inline int aruco_dict_number = 2; // DICT_4X4_50=0, DICT_4X4_100=1, DICT_4X4_250=2, DICT_4X4_1000=3, DICT_5X5_50=4, DICT_5X5_100=5...
inline int aruco_marker_id = 11;
inline int aruco_marker_pixels = 500; // Only used for generating markers
inline float markerLength = 0.046f; // Marker length in meters
inline extern std::string f_markers_location = "../f_markers/";
inline extern std::string f_marker_name = "marker_" + std::to_string(aruco_marker_id)
      + "_dict_" + std::to_string(aruco_dict_number) + "_size_" + std::to_string(aruco_marker_pixels) + ".png";

// Post-processing filters for the depth frame
// Threshold filter, Temporal filter, Hole filling filter
inline extern rs2::threshold_filter threshold_filter = rs2::threshold_filter();
inline extern rs2::temporal_filter temporal_filter = rs2::temporal_filter();
inline extern rs2::hole_filling_filter hole_filling_filter = rs2::hole_filling_filter();
// Values for the filters
inline extern float depth_filter_smooth_alpha = 0.25f; // Smoothing factor 0.25
inline extern float depth_filter_smooth_delta = 60; // Delta value 60
inline extern float depth_filter_temporal_holes_fill = 6.0f; // Persistency index (highest = 8) [0-8]
inline extern float depth_filter_holes_fill = 2.0f;// 2 = fill from the farthest pixel

// This function takes the width and height of a depth image and returns the x and y start and stop points for cropping
// The start and stop points are the absolute vales of the start position and stop position of the crop
// E.g. start_x = 150, stop_x = 450, start_y = 100, stop_y = 350 for a 640x480 image
// The percentage is the value of what the image should be reduced to (i.e. 40% of the original image)
inline std::tuple<int, int, int, int> get_crop_points(const int width, const int height, const int percentage) {
  // Check to see if value is between 0 and 100
  if (percentage < 0 || percentage > 100) {
    throw std::invalid_argument("Percentage must be between 0 and 100");
  }
  int x_start = width * (100 - percentage) / 200;
  int y_start = height * (100 - percentage) / 200;
  int x_stop = width * (100 + percentage) / 200;
  int y_stop = height * (100 + percentage) / 200;
  return std::make_tuple(x_start, x_stop, y_start, y_stop);
}

// Initialise the filters for the depth frame
inline void initialise_filters() {
  threshold_filter.set_option(RS2_OPTION_MIN_DISTANCE, depth_filter_min_distance);
  threshold_filter.set_option(RS2_OPTION_MAX_DISTANCE, depth_filter_max_distance);

  temporal_filter.set_option(RS2_OPTION_FILTER_SMOOTH_ALPHA, depth_filter_smooth_alpha);
  temporal_filter.set_option(RS2_OPTION_FILTER_SMOOTH_DELTA, depth_filter_smooth_delta);
  temporal_filter.set_option(RS2_OPTION_HOLES_FILL, depth_filter_temporal_holes_fill);

  hole_filling_filter.set_option(RS2_OPTION_HOLES_FILL, depth_filter_holes_fill);
}

// Apply post-processing filters to the depth frame
inline rs2::depth_frame apply_depth_post_processing_filters(rs2::depth_frame depth) {
  depth = threshold_filter.process(depth);
  depth = temporal_filter.process(depth);
  //depth = hole_filling_filter.process(depth);
  return depth;
}

// Convert the depth frame to a PCL point cloud
// Allows for optional cropping of the image if crop is set to true
// The crop is centered around the center of the image
inline pcl::PointCloud<pcl::PointXYZ>::Ptr depthFrameToPointCloud(const rs2::depth_frame& depth, const bool crop = false,
  const bool debugging = false) {
  // Convert depth frame to a PCL point cloud
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
  const rs2::pointcloud rs_cloud;
  const rs2::points points = rs_cloud.calculate(depth);
  const auto stream_profile = points.get_profile().as<rs2::video_stream_profile>();
  const int width = stream_profile.width();
  const int height = stream_profile.height();
  cloud->width = width;
  cloud->height = height;
  cloud->is_dense = false;
  cloud->points.resize(points.size());

  // Initialise start and stop to default values
  int x_start = 0, x_stop = width, y_start = 0, y_stop = height;

  // We want to crop the image to focus only on the center
  if (crop) {
    const auto [x_start_crop, x_stop_crop, y_start_crop, y_stop_crop]
                = get_crop_points(width, height, image_reduced_to_percentage);
    // Update the start and stop values
    x_start = x_start_crop;
    x_stop = x_stop_crop;
    y_start = y_start_crop;
    y_stop = y_stop_crop;
  }

  // Extract the points
  const auto vertices = points.get_vertices();
  int i = 0;
  for (int y = y_start; y < y_stop; y++) {
    for (int x = x_start; x < x_stop; x++, i++) {
      const int index = y * width + x;
      cloud->points[i].x = vertices[index].x;
      cloud->points[i].y = vertices[index].y;
      cloud->points[i].z = vertices[index].z;
    }
  }
  // set the size of the point cloud
  cloud->width = x_stop - x_start;
  cloud->height = y_stop - y_start;
  cloud->is_dense = false;

  if (debugging) {
    std::cout << "PointCloud captured from Realsense camera has: " << cloud->size() << " data points." << std::endl;
  }
  return cloud;
}

// This is the Cluster Extraction code that is common to both the test and the main code
// It extracts clusters from the point cloud
inline void filterAndSegmentPointCloud(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud,
  pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud_filtered, std::vector<pcl::PointIndices>& cluster_indices,
  const bool debugging = false) {

  // Create the filtering object: down-sample the dataset using a leaf size of 1cm
  pcl::VoxelGrid<pcl::PointXYZ> vg;
  vg.setInputCloud(cloud);
  vg.setLeafSize(leaf_size, leaf_size, leaf_size);
  vg.filter(*cloud_filtered);
  // std::cout << "PointCloud after filtering has: " << cloud_filtered->size() << " data points." << std::endl;

  // Create the segmentation object for the planar model and set all the parameters
  pcl::SACSegmentation<pcl::PointXYZ> seg;
  pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
  pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_plane(new pcl::PointCloud<pcl::PointXYZ>());
  seg.setOptimizeCoefficients(true);
  seg.setModelType(pcl::SACMODEL_PLANE);
  seg.setMethodType(pcl::SAC_RANSAC);
  seg.setMaxIterations(100); // 100 default
  seg.setDistanceThreshold(segment_distance_threshold); // 0.02 default

  // Additional parameters to improve sample quality
  seg.setProbability(segment_probability);
  seg.setRadiusLimits(segment_radius_min, segment_radius_max);

  int filter_count = 0;
  int nr_points = static_cast<int>(cloud_filtered->size());

  // Create the filtering object
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_f(new pcl::PointCloud<pcl::PointXYZ>);
  while (cloud_filtered->size () > filter_cluster_to_x_percent * nr_points) {
  // while (filter_count < 1) {
    // Segment the largest planar component from the remaining cloud
    seg.setInputCloud(cloud_filtered);
    seg.segment(*inliers, *coefficients);
    if (inliers->indices.empty()) {
      std::cout << "Could not estimate a planar model for the given dataset." << std::endl;
      return;
    }

    // Extract the planar inliers from the input cloud
    pcl::ExtractIndices<pcl::PointXYZ> extract;
    extract.setInputCloud(cloud_filtered);
    extract.setIndices(inliers);
    extract.setNegative(false);

    // Get the points associated with the planar surface
    extract.filter(*cloud_plane);
    if (debugging) {
      std::cout << "PointCloud representing the planar component: " << cloud_plane->size() << " data points." << std::endl;
    }

    // Remove the planar inliers, extract the rest
    extract.setNegative(true);
    extract.filter(*cloud_f);
    *cloud_filtered = *cloud_f;

    filter_count++;
  }

  // Creating the KdTree object for the search method of the extraction
  pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>);
  tree->setInputCloud(cloud_filtered);

  pcl::EuclideanClusterExtraction<pcl::PointXYZ> ec;
  ec.setClusterTolerance(cluster_tolerance);
  ec.setMinClusterSize(min_cluster_size);
  ec.setMaxClusterSize(max_cluster_size);
  ec.setSearchMethod(tree);
  ec.setInputCloud(cloud_filtered);
  ec.extract(cluster_indices);

  if (debugging)
  {
    cout << "Finished trying to find clusters " << endl;
  }
}


[[noreturn]] inline void processPointCloud(const rs2::pipeline& pipeline,
  pcl::PointCloud<pcl::PointXYZ>::Ptr& output_stream_cloud, const std::shared_ptr<std::vector<pcl::PointIndices>>& cluster_indices,
  const bool& debug, std::mutex& mtx, std::condition_variable& cv, bool& ready)
{
  while (true) {
    auto frames = pipeline.wait_for_frames();
    const auto depth = frames.get_depth_frame();
    rs2::depth_frame filtered_depth = apply_depth_post_processing_filters(depth);
    auto cloud = depthFrameToPointCloud(filtered_depth, true, debug);
    pcl::PointCloud<pcl::PointXYZ>::Ptr filtered_cloud_clusters(new pcl::PointCloud<pcl::PointXYZ>);
    std::vector<pcl::PointIndices> cluster_indices_local;
    filterAndSegmentPointCloud(cloud, filtered_cloud_clusters, cluster_indices_local, debug);

    if (cluster_indices_local.empty() && debug) {
      std::cout << "No clusters found." << std::endl;
      continue;
    }

    {
      std::lock_guard<std::mutex> lock(mtx);
      output_stream_cloud = filtered_cloud_clusters;
      *cluster_indices = cluster_indices_local;
      ready = true;
    }
    cv.notify_one();
  }
}

// This function estimates VFH signatures of an input point cloud
// Input: File path
// Output: VFH signature of the object
inline void estimate_VFH(const pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud, pcl::PointCloud <pcl::VFHSignature308> &signature)
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

// This function checks if the file already contains a VFH signature
// Input: File path
// Output: true if file contains VFH signature, false if file does not contain VFH signature
inline bool checkVFH(const boost::filesystem::path &path)
{
    try
  {
    // Read point cloud header
    pcl::PCLPointCloud2 cloud;
    int version;
    Eigen::Vector4f origin;
    Eigen::Quaternionf orientation;
    pcl::PCDReader r;
    int type; unsigned int idx;
    r.readHeader (path.string (), cloud, origin, orientation, version, type, idx);

 // Check if there is VFH field in the cloud header
    if (const int vfh_idx = pcl::getFieldIndex(cloud, "vfh"); vfh_idx == -1)
      return (false);
    if (static_cast<int>(cloud.width) * cloud.height != 1)
      return (false);
  }
  catch (const pcl::InvalidConversionException&)
  {
    return (false);
  }
  return (true);
}

// This function loads VFH signature histogram into a vfh model
// Input: File path to histogram
// Output: A boolean that returns true if the histogram is loaded successfully and a vfh_model data that holds the histogram information
inline bool load_vfh_histogram (const boost::filesystem::path &path, vfh_model &vfh)
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

// This gets PCD file names from a directory and passes them to histogram loader one by one
// then pushes the results into a vfh_model vector to keep them in a data storage
// Input: Model data set file path
// Output: A vfh_model vector that contains all VFH signature information
inline void load_vfh_model_data(const boost::filesystem::path &base_dir, std::vector<vfh_model> &models,
  std::vector<std::string> &model_files) {

  if (!exists(base_dir) || !is_directory(base_dir)) {
    return;
  }

  std::stack<boost::filesystem::path> directories;
  directories.push(base_dir);

  while (!directories.empty()) {
    boost::filesystem::path current_dir = directories.top();
    directories.pop();

    int vfh_count = 0;

    for (boost::filesystem::directory_iterator i(current_dir); i != boost::filesystem::directory_iterator(); ++i) {
      if (boost::filesystem::is_directory(i->status())) {
        directories.push(i->path());
      } else if (boost::filesystem::is_regular_file(i->status()) && boost::filesystem::extension(i->path()) == ".pcd") {
        vfh_model m;
        if (load_vfh_histogram(i->path().string(), m)) {
          models.push_back(m);
          model_files.push_back(i->path().string()); // Store the file name
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

// A small helper to replace a deprecated Boost Function
inline void replace_last(std::string& str, const std::string& from, const std::string& to) {
    std::size_t pos = str.rfind(from);
    if (pos != std::string::npos) {
        str.replace(pos, from.length(), to);
    }
}

// This function visualizes given k point cloud arguments on a PCL_Viewer
// It shows distances of candidates from the query object at the left bottom corner of each window
// If the distance is smaller than a given threshold then the distances are shown in green, else they are shown in red
// Inputs: Main function arguments, candidate count (k), threshold value, vfh_model vector that contains VFH signatures
// from the data set, indices of candidates and distances of candidates
inline void visualise(int argc, char** argv, int k, double thresh, std::vector<vfh_model> models, flann::Matrix<int> k_indices, flann::Matrix<float> k_distances)
{
  // Load the results
  pcl::visualization::PCLVisualizer p (argc, argv, "VFH Cluster Classifier");
  int y_s = static_cast<int>(floor(sqrt(static_cast<double>(k))));
  int x_s = y_s + static_cast<int>(ceil((k / static_cast<double>(y_s)) - y_s));
  auto x_step = 1 / static_cast<double>(x_s);
  auto y_step = 1 / static_cast<double>(y_s);
  pcl::console::print_highlight ("Preparing to load ");
  pcl::console::print_value ("%d", k);
  pcl::console::print_info (" files (");
  pcl::console::print_value ("%d", x_s);
  pcl::console::print_info ("x");
  pcl::console::print_value ("%d", y_s);
  pcl::console::print_info (" / ");
  pcl::console::print_value ("%f", x_step);
  pcl::console::print_info ("x");
  pcl::console::print_value ("%f", y_step);
  pcl::console::print_info (")\n");

  int viewport = 0, l = 0, m = 0;
  for (int i = 0; i < k; ++i)
  {
    std::string cloud_name = models.at (k_indices[0][i]).first;
    replace_last (cloud_name, "_vfh", "");

    p.createViewPort (l * x_step, m * y_step, (l + 1) * x_step, (m + 1) * y_step, viewport);
    l++;
    if (l >= x_s)
    {
      l = 0;
      m++;
    }

    pcl::PCLPointCloud2 cloud;
    pcl::console::print_highlight (stderr, "Loading "); pcl::console::print_value (stderr, "%s ", cloud_name.c_str ());
    if (pcl::io::loadPCDFile (cloud_name, cloud) == -1)
      break;

    // Convert from blob to PointCloud
    pcl::PointCloud<pcl::PointXYZ> cloud_xyz;
    pcl::fromPCLPointCloud2 (cloud, cloud_xyz);

    if (cloud_xyz.points.empty())
      break;

    pcl::console::print_info ("[done, ");
    pcl::console::print_value ("%d", static_cast<int>(cloud_xyz.points.size()));
    pcl::console::print_info (" points]\n");
    pcl::console::print_info ("Available dimensions: ");
    pcl::console::print_value ("%s\n", pcl::getFieldsList (cloud).c_str ());

    // Demean the cloud
    Eigen::Vector4f centroid;
    pcl::compute3DCentroid (cloud_xyz, centroid);
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_xyz_demean (new pcl::PointCloud<pcl::PointXYZ>);
    pcl::demeanPointCloud<pcl::PointXYZ> (cloud_xyz, centroid, *cloud_xyz_demean);
    // Add to renderer*
    p.addPointCloud (cloud_xyz_demean, cloud_name, viewport);

    // Check if the model found is within our inlier tolerance
    std::stringstream ss;
    ss << k_distances[0][i];
    if (k_distances[0][i] > thresh)
    {
      p.addText (ss.str (), 20, 30, 1, 0, 0, ss.str (), viewport);  // display the text with red

      // Create a red line
      pcl::PointXYZ min_p, max_p;
      pcl::getMinMax3D (*cloud_xyz_demean, min_p, max_p);
      std::stringstream line_name;
      line_name << "line_" << i;
      p.addLine (min_p, max_p, 1, 0, 0, line_name.str (), viewport);
      p.setShapeRenderingProperties (pcl::visualization::PCL_VISUALIZER_LINE_WIDTH, 5, line_name.str (), viewport);
    }
    else
      p.addText (ss.str (), 20, 30, 0, 1, 0, ss.str (), viewport);

    // Increase the font size for the score*
    p.setShapeRenderingProperties (pcl::visualization::PCL_VISUALIZER_FONT_SIZE, 18, ss.str (), viewport);

    // Add the cluster name
    p.addText (cloud_name, 20, 10, cloud_name, viewport);
  }
  // Add coordinate systems to all viewports
  p.addCoordinateSystem (0.1, "global", 0);

  p.spin ();
}

// This function finds the k nearest neighbors of the query object and returns their indices in the K-d tree and distances from the query object
// Inputs: K-d tree, query object model, desired candidate count (k)
// Outputs: K-d tree indices of candidates, Distances of candidates from the query object
inline void nearestKSearch (const flann::Index<flann::ChiSquareDistance<float> > &index, const vfh_model &model,
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

// Function to show the point clouds with increased point size
inline void showPointClouds(const std::vector<std::string>& created_files, const bool ignore_vfh = false) {
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

// Function to get all files matching a pattern
inline std::vector<std::string> globFiles(const std::string& pattern) {
  glob_t glob_result;
  glob(pattern.c_str(), GLOB_TILDE, nullptr, &glob_result);
  std::vector<std::string> files;
  for (unsigned int i = 0; i < glob_result.gl_pathc; ++i) {
    files.emplace_back(glob_result.gl_pathv[i]);
  }
  globfree(&glob_result);
  return files;
}

/////////////// BELOW IS THE POSE ESTIMATION SPECIFIC CODE ///////////////
class PoseManager {
public:
  PoseManager() : stored_pose(Eigen::Matrix4f::Identity()) {}

  void setPose(const Eigen::Matrix4f& pose) {
    stored_pose = pose;
  }

  void updatePose(const Eigen::Matrix4f& pose) {
    const float distance = (stored_pose.block<3, 1>(0, 3) - pose.block<3, 1>(0, 3)).norm();
    const float update_factor = calculateUpdateFactor(distance);

    stored_pose = update_factor * pose + (1.0f - update_factor) * stored_pose;
  }

  [[nodiscard]] Eigen::Matrix4f getPose() const {
    return stored_pose;
  }

private:
  Eigen::Matrix4f stored_pose;

  static float calculateUpdateFactor(const float distance) {
    if (distance < start_scale_distance) {
      return 1.0f;
    } else if (distance > stop_scale_distance) {
      return 0.05f;
    } else {
      const auto scale = static_cast<float>(stability_factor);
      const float normalized_distance = (distance - start_scale_distance) / (stop_scale_distance - start_scale_distance);
      return 1.0f - std::pow(normalized_distance, scale) * (1.0f - 0.05f);
    }
  }
};

// Helper function to transform a point using the pose matrix
inline cv::Point3f transformPoint(const Eigen::Matrix4f& pose, const cv::Point3f& point) {
  const Eigen::Vector4f point_homogeneous(point.x, point.y, point.z, 1.0);
  Eigen::Vector4f point_transformed = pose * point_homogeneous;
  return {point_transformed.x(), point_transformed.y(), point_transformed.z()};
}

// Helper function to project a 3D point to 2D image point
inline cv::Point2f projectPoint(const cv::Point3f& point, const rs2_intrinsics& intrinsics) {
  float x_2d = intrinsics.fx * point.x / point.z + intrinsics.ppx;
  float y_2d = intrinsics.fy * point.y / point.z + intrinsics.ppy;
  return {x_2d, y_2d};
}

// Function to estimate the pose of the object.
inline Eigen::Matrix4f estimatePose(const pcl::PointCloud<pcl::PointXYZ>::Ptr& scene_cloud,
  const pcl::PointCloud<pcl::PointXYZ>::Ptr& model_cloud) {

  pcl::IterativeClosestPoint<pcl::PointXYZ, pcl::PointXYZ> icp;
  icp.setInputSource(model_cloud);
  icp.setInputTarget(scene_cloud);
  icp.setMaximumIterations(icp_max_iterations);
  icp.setTransformationEpsilon(icp_transformation_epsilon);
  icp.setEuclideanFitnessEpsilon(icp_euclidean_fitness_epsilon);
  pcl::PointCloud<pcl::PointXYZ> Final;
  icp.align(Final);

  if (icp.hasConverged()) {
    return icp.getFinalTransformation();
  } else {
    return Eigen::Matrix4f::Identity();
  }
}

// Function to display the coordinate system on top of the video feed
inline void displayCoordinateSystem(const Eigen::Matrix4f& pose, cv::Mat& frame, const rs2_intrinsics& intrinsics) {
  // Define the origin and axes in the object coordinate system
  const cv::Point3f origin(0, 0, 0);
  const cv::Point3f x_axis(0.1, 0, 0);
  const cv::Point3f y_axis(0, 0.1, 0);
  const cv::Point3f z_axis(0, 0, 0.1);

  // Transform the axes to the camera coordinate system
  const cv::Point3f origin_transformed = transformPoint(pose, origin);
  const cv::Point3f x_axis_transformed = transformPoint(pose, x_axis);
  const cv::Point3f y_axis_transformed = transformPoint(pose, y_axis);
  const cv::Point3f z_axis_transformed = transformPoint(pose, z_axis);

  // Project the 3D points to 2D image points
  const cv::Point2f origin_2d = projectPoint(origin_transformed, intrinsics);
  const cv::Point2f x_axis_2d = projectPoint(x_axis_transformed, intrinsics);
  const cv::Point2f y_axis_2d = projectPoint(y_axis_transformed, intrinsics);
  const cv::Point2f z_axis_2d = projectPoint(z_axis_transformed, intrinsics);

  // Draw the coordinate system on the frame
  cv::line(frame, origin_2d, x_axis_2d, cv::Scalar(0, 0, 255), 3); // X-axis in red
  cv::line(frame, origin_2d, y_axis_2d, cv::Scalar(0, 255, 0), 3); // Y-axis in green
  cv::line(frame, origin_2d, z_axis_2d, cv::Scalar(255, 0, 0), 3); // Z-axis in blue
}

// This function takes a point cloud and sets the origin to the center of the point cloud
// It also sets the axes to the principal axes of the point cloud, currently y = up, z = forward/front of the object
// 0 degrees is defined as directly facing the camera

// DEPRECATED: This function is not used anymore - it was replaced by setGlobalOriginAndAxes
inline void getPointCloudOriginAndAxes(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud_cluster, Eigen::Matrix4f& pose,
  const float object_facing_angle, const float calibration_angle_offset, const rs2_vector& accel_data) {

  // Compute the centroid of the point cloud
  Eigen::Vector4f centroid;
  compute3DCentroid(*cloud_cluster, centroid);

  // Translate the point cloud to the origin (centroid)
  Eigen::Affine3f transform = Eigen::Affine3f::Identity();
  transform.translation() << -centroid[0], -centroid[1], -centroid[2];
  transformPointCloud(*cloud_cluster, *cloud_cluster, transform);

  Eigen::Vector3f gravity_direction(accel_data.x, accel_data.y, accel_data.z);
  gravity_direction.normalize();

  // Calculate the angle between the y-axis and the gravity direction
  const Eigen::Vector3f y_axis = Eigen::Vector3f::UnitY();
  const float angle = acos(y_axis.dot(gravity_direction));

  // Subtract the angle from the gravity direction
  const auto rotation_quart = Eigen::Quaternionf(Eigen::AngleAxisf(-angle, y_axis.cross(gravity_direction).normalized()));
  const Eigen::Matrix3f y_align_rotation = rotation_quart.toRotationMatrix();

  // Apply the opposite of the resulting transformation to the point cloud (to point away from gravity)
  pose.block<3, 3>(0, 0) = -y_align_rotation;

  // TODO: Figure out this object rotation a bit better and more efficiently
  // Now that the y direction is set, we can rotate the object to align with zero degrees
  Eigen::Matrix3f object_rotation; // Some - and + values to get 0 to be approximately directly away from the camera
  object_rotation = Eigen::AngleAxisf(M_PI * (object_facing_angle+calibration_angle_offset) / 180.0f, Eigen::Vector3f::UnitY());
  pose.block<3, 3>(0, 0) = object_rotation * pose.block<3, 3>(0, 0);

  // The x-axis can be set to the cross product of the y-axis and z-axis
  pose.block<3, 1>(0, 0) = pose.block<3, 1>(0, 1).cross(pose.block<3, 1>(0, 2));
}

// This function transforms the camera coordinate system to the global coordinate system
// Is it currently the camera's position rectified (y pointing directly up, z is the same direction as/parallel to the front of the object)
// The global coordinate system is defined as the camera's position when the program starts
inline void setGlobalOriginAndAxes(Eigen::Matrix4f& pose, const rs2_vector& accel_data,
  const float object_facing_angle = 180, const float calibration_angle_offset = 0) {

  // Get the gravity direction from the accelerometer data
  Eigen::Vector3f gravity_direction(accel_data.x, accel_data.y, accel_data.z);
  gravity_direction.normalize();

  // Calculate the angle between the y-axis and the gravity direction
  const Eigen::Vector3f y_axis = Eigen::Vector3f::UnitY();
  const float angle = acos(y_axis.dot(gravity_direction));

  // Subtract the angle from the gravity direction
  const auto rotation_quart = Eigen::Quaternionf(Eigen::AngleAxisf(-angle, y_axis.cross(gravity_direction).normalized()));
  pose.block<3, 3>(0, 0)  = -rotation_quart.toRotationMatrix();

  /*// Apply the opposite of the resulting transformation to the point cloud (to point away from gravity)
  pose.block<3, 3>(0, 0) = -y_align_rotation;*/

  // Now that the y direction is set, we can rotate the coordinate to align with the 'forward' direction
  Eigen::Matrix3f object_rotation; // Some - and + values to get 0 to be approximately directly away from the camera
  object_rotation = Eigen::AngleAxisf(M_PI * (object_facing_angle+calibration_angle_offset) / 180.0f, Eigen::Vector3f::UnitY());
  pose.block<3, 3>(0, 0) = object_rotation * pose.block<3, 3>(0, 0);

  // The x-axis can be set to the cross product of the y-axis and z-axis
  pose.block<3, 1>(0, 0) = pose.block<3, 1>(0, 1).cross(pose.block<3, 1>(0, 2));
}