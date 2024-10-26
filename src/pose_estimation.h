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

#include <opencv2/opencv.hpp>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/segmentation/sac_segmentation.h>

// Define vfh_model type for storing data file name and histogram data
typedef std::pair<std::string, std::vector<float> > vfh_model;

// Declare global variables that all functions in the .cpp files can access
inline extern const int image_reduced_to_percentage = 60;
inline extern const float depth_filter_min_distance = 0.5f;
inline extern const float depth_filter_max_distance = 1.0f;

// Global camera config params
// 848x480 resolution, 15 frames per second is optimal for Realsense D455
inline extern const int cam_res_width = 848;
inline extern const int cam_res_height = 480;
inline extern const int cam_fps = 15;

// Directory for reference models
inline extern std::string model_directory = "../lab_data/";

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

// Apply global threshold filter to the depth frame
inline rs2::depth_frame apply_threshold_filter(rs2::depth_frame depth) {
  const rs2::threshold_filter threshold_filter;
  threshold_filter.set_option(RS2_OPTION_MIN_DISTANCE, depth_filter_min_distance);
  threshold_filter.set_option(RS2_OPTION_MAX_DISTANCE, depth_filter_max_distance);
  return threshold_filter.process(std::move(depth));
}

// Convert the depth frame to a PCL point cloud
// Allows for optional cropping of the image if crop is set to true
// The crop is centered around the center of the image
inline pcl::PointCloud<pcl::PointXYZ>::Ptr depthFrameToPointCloud(rs2::depth_frame depth,
  const bool filter = false, const bool crop = false) {

  if (filter) {
    // Filter the depth values of the depth frame
    depth = apply_threshold_filter(depth);
  }

  // Convert the depth frame to a PCL point cloud
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
      cloud->points[i].y = -vertices[index].y;
      cloud->points[i].z = -vertices[index].z;
    }
  }
  // set the size of the point cloud
  cloud->width = x_stop - x_start;
  cloud->height = y_stop - y_start;
  cloud->is_dense = false;
  return cloud;
}

inline void stream_point_cloud_show_depth_map(rs2::pipeline pipe, pcl::PointCloud<pcl::PointXYZ>::Ptr &output_stream_cloud,
  std::mutex &mtx, std::condition_variable &cv, bool &ready) {

  // Add a stream with its parameters
  rs2::config config;
  config.enable_stream(RS2_STREAM_DEPTH, cam_res_width, cam_res_height, RS2_FORMAT_Z16, cam_fps);

  // Instruct pipeline to start streaming with the requested configuration
  pipe.start(config);

  // Camera warmup - dropping several first frames to let auto-exposure stabilize
  for (int i = 0; i < 100; i++) {
      // Wait for all configured streams to produce a frame
      auto frames = pipe.wait_for_frames(); // purposely does nothing with the frames
  }

  while (true) {
    // Wait for the next set of frames from the camera
    auto frames = pipe.wait_for_frames();

    // Get a frame from the depth stream
    auto depth = frames.get_depth_frame();

    auto cloud = depthFrameToPointCloud(depth, true, true);

    // Everything past here is for visualisation only
    // Same threshold filter as the one that was applied for the point cloud
    // Same cropping as the one that was applied for the point cloud
    auto filtered_depth = apply_threshold_filter(depth);

    // Get width and height of the depth frame
    const auto width = filtered_depth.get_width();
    const auto height = filtered_depth.get_height();

    // Create OpenCV matrix of size (w,h) from the depth frame
    cv::Mat depth_image(cv::Size(width, height), CV_16UC1, const_cast<void*>(filtered_depth.get_data()), cv::Mat::AUTO_STEP);

    // Get crop points using the get_crop_points function
    const auto [x_start, x_stop, y_start, y_stop]
                = get_crop_points(width, height, image_reduced_to_percentage);
    cv::Mat cropped_depth_image = depth_image(cv::Rect(x_start, y_start, x_stop - x_start, y_stop - y_start));

    // Convert the depth image to CV_8UC1
    cv::Mat depth_image_8u;
    cropped_depth_image.convertTo(depth_image_8u, CV_8UC1, 255.0 / 10000); // Scale the depth values to 8-bit

    // Apply colormap on depth image
    cv::Mat depth_colormap;
    cv::applyColorMap(depth_image_8u, depth_colormap, cv::COLORMAP_JET);

    // Lock the mutex and update the shared output_depth_map
    {
        std::lock_guard<std::mutex> lock(mtx);
        output_stream_cloud = cloud;
        ready = true;
    }
    cv.notify_one();

    // Display the depth map
    cv::imshow("Depth Map Stream", depth_colormap);
    if (cv::waitKey(1) >= 0) {
        break;
    }
  }
  // Stop the pipeline
  pipe.stop();
  cv::destroyAllWindows();
}

// This function estimates VFH signatures of an input point cloud 
// Input: File path
// Output: VFH signature of the object
inline void estimate_VFH(const pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud, pcl::PointCloud <pcl::VFHSignature308> &signature)
{
  pcl::PointCloud<pcl::Normal>::Ptr normals (new pcl::PointCloud<pcl::Normal> ());
  
  // Estimate point cloud normals
  pcl::search::KdTree<pcl::PointXYZ>::Ptr normalTree (new pcl::search::KdTree<pcl::PointXYZ> ());
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
// Input: File path
// Output: A boolean that returns true if the histogram is loaded successfully and a vfh_model data that holds the histogram information
inline bool loadHist (const boost::filesystem::path &path, vfh_model &vfh)
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
// Input: Training data set file path
// Output: A vfh_model vector that contains all VFH signature information
inline void loadData(const boost::filesystem::path &base_dir, std::vector<vfh_model> &models) {
  if (!boost::filesystem::exists(base_dir) || !boost::filesystem::is_directory(base_dir)) {
    return;
  }

  std::stack<boost::filesystem::path> directories;
  directories.push(base_dir);

  while (!directories.empty()) {
    boost::filesystem::path current_dir = directories.top();
    directories.pop();

    for (boost::filesystem::directory_iterator i(current_dir); i != boost::filesystem::directory_iterator(); ++i) {
      if (boost::filesystem::is_directory(i->status())) {
        directories.push(i->path());
        std::stringstream ss;
        ss << i->path();
        pcl::console::print_highlight("Loading %s (%lu models loaded so far).\n", ss.str().c_str(), (unsigned long)models.size());
      } else if (boost::filesystem::is_regular_file(i->status()) && boost::filesystem::extension(i->path()) == ".pcd") {
        vfh_model m;
        if (loadHist(i->path().string(), m)) {
          models.push_back(m);
          std::cout << m.first << std::endl;
        }
      }
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
inline void visualize(int argc, char** argv, int k, double thresh, std::vector<vfh_model> models, flann::Matrix<int> k_indices, flann::Matrix<float> k_distances)
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
inline void showPointClouds(const std::vector<std::string>& created_files) {
  const pcl::visualization::PCLVisualizer::Ptr viewer(new pcl::visualization::PCLVisualizer("3D Viewer"));
  viewer->setBackgroundColor(0, 0, 0);

  for (const auto& file : created_files) {
    if (file.substr(file.find_last_of(".") + 1) != "pcd" || file.find("_vfh.pcd") != std::string::npos) {
      continue; // Skip non-pcd files and _vfh.pcd files
    }
    std::cout << "Loading .pcd file for visualisation: " << file << std::endl;
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
    if (pcl::io::loadPCDFile(file, *cloud) == -1) {
      PCL_ERROR("Couldn't read file %s \n", file.c_str());
      continue;
    }
    viewer->addPointCloud<pcl::PointXYZ>(cloud, file);
    viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 5, file); // Increase point size
  }

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

// This is the Cluster Extraction code that is common to both the test and the main code
// It extracts clusters from the point cloud
inline void filterAndSegmentPointCloud(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud, pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud_filtered,
  pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud_f, std::vector<pcl::PointIndices>& cluster_indices) {

  // Create the filtering object: down-sample the dataset using a leaf size of 1cm
  pcl::VoxelGrid<pcl::PointXYZ> vg;
  vg.setInputCloud(cloud);
  vg.setLeafSize(0.006f, 0.006f, 0.006f); // 0.01f default
  vg.filter(*cloud_filtered);
  std::cout << "PointCloud after filtering has: " << cloud_filtered->size() << " data points." << std::endl;

  // Create the segmentation object for the planar model and set all the parameters
  pcl::SACSegmentation<pcl::PointXYZ> seg;
  pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
  pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_plane(new pcl::PointCloud<pcl::PointXYZ>());
  seg.setOptimizeCoefficients(true);
  seg.setModelType(pcl::SACMODEL_PLANE);
  seg.setMethodType(pcl::SAC_RANSAC);
  seg.setMaxIterations(100); // 100 default
  seg.setDistanceThreshold(0.02); // 0.02 default

  int filter_count = 0;
  int nr_points = static_cast<int>(cloud_filtered->size());


  while (cloud_filtered->size () > 0.3 * nr_points) {
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
    std::cout << "PointCloud representing the planar component: " << cloud_plane->size() << " data points." << std::endl;

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
  ec.setClusterTolerance(0.01); // 2cm == 0.02 default
  ec.setMinClusterSize(300); // 100 default
  ec.setMaxClusterSize(1000); // 25000 default
  ec.setSearchMethod(tree);
  ec.setInputCloud(cloud_filtered);
  ec.extract(cluster_indices);
}