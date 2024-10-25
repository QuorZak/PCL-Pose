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
#include <condition_variable>

#include <opencv2/opencv.hpp>

// Define vfh_model type for storing data file name and histogram data
typedef std::pair<std::string, std::vector<float> > vfh_model;

// This function streams the depth map from the Realsense camera through OpenCV
inline void streamDepthMap(rs2::pipeline pipe, rs2::config cfg, cv::Mat &output_depth_map, std::mutex &mtx, std::condition_variable &cv, bool &ready) {
    // Create a threshold filter
    rs2::threshold_filter threshold_filter;
    threshold_filter.set_option(RS2_OPTION_MIN_DISTANCE, 0.2f);
    threshold_filter.set_option(RS2_OPTION_MAX_DISTANCE, 1.2f);


    // We will want to crop the image to focus only on the center of the image (50% in x and y)
    // And also slightly lower in the y direction (10%)
    constexpr int overall_reduction_percentage = 40;
    constexpr int y_downshift_percentage = 10;

    // Implement the percentages
    constexpr int x_offset = 640 * overall_reduction_percentage / 100 / 2;
    constexpr int y_offset = 480 * overall_reduction_percentage / 100 / 2 + 480 * y_downshift_percentage / 100;
    constexpr int crop_width = 640 * overall_reduction_percentage / 100;
    constexpr int crop_height = 480 * overall_reduction_percentage / 100;

    // Add a stream with its parameters
    cfg.enable_stream(RS2_STREAM_DEPTH, 640, 480, RS2_FORMAT_Z16, 15);

    // Instruct pipeline to start streaming with the requested configuration
    pipe.start(cfg);

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
        // filter the depth frame
        depth = threshold_filter.process(depth);

        // Create OpenCV matrix of size (w,h) from the depth frame
        cv::Mat depth_image(cv::Size(640, 480), CV_16UC1, const_cast<void*>(depth.get_data()), cv::Mat::AUTO_STEP);
        cv::Mat cropped_depth_image = depth_image(cv::Rect(x_offset, y_offset, crop_width, crop_height));

        // Convert the depth image to CV_8UC1
        cv::Mat depth_image_8u;
        cropped_depth_image.convertTo(depth_image_8u, CV_8UC1, 255.0 / 10000); // Scale the depth values to 8-bit

        // Apply colormap on depth image
        cv::Mat depth_colormap;
        cv::applyColorMap(depth_image_8u, depth_colormap, cv::COLORMAP_JET);

        // Lock the mutex and update the shared output_depth_map
        {
            std::lock_guard<std::mutex> lock(mtx);
            output_depth_map = depth_colormap.clone();
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
inline void loadData(const boost::filesystem::path &base_dir, std::vector<vfh_model> &models)
{
  if (!boost::filesystem::exists (base_dir) && !boost::filesystem::is_directory (base_dir))
    return;
  else
  {
	// Iterate through the data set directory to read VFH signatures
  	for (boost::filesystem::directory_iterator i (base_dir); i != boost::filesystem::directory_iterator (); ++i)
	{
	  // If read path is a directory, then print path name on console and call data loader again
	  if (boost::filesystem::is_directory (i->status ()))
	  {
	  	std::stringstream ss;
		ss << i->path ();
		pcl::console::print_highlight ("Loading %s (%lu models loaded so far).\n", ss.str ().c_str (), (unsigned long)models.size ());
	    loadData (i->path (), models);
	  }
	  // If read path is a file with *.pcd extension, then check if it contains VFH signature
	  // If not, then estimate VFH signature of it
	  // Either way, pass the file to histogram loader since it can differentiate between files with VFH signatures and files without them
	  if (boost::filesystem::is_regular_file (i->status ()) && boost::filesystem::extension (i->path ()) == ".pcd")
	  {
		  vfh_model m;
		  std::string str = i->path ().string();
		  if (loadHist (i->path ().string(), m))
		  {
		  	models.push_back (m);	
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
