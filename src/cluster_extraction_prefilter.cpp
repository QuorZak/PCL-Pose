#include <pcl/ModelCoefficients.h>
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/search/kdtree.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/segmentation/extract_clusters.h>
#include <iomanip> // for setw, setfill

#include <pose_estimation.h>

int
main ()
{
  const std::string output_folder = "../lab_data/";

  rs2::pipeline pipeline;
  rs2::frame depth;
  rs2::pointcloud rs_cloud;
  rs2::points points;

  rs2::pipeline pipe;
  rs2::config cfg;
  cv::Mat output_depth_map;
  cv::Mat depth_map;
  std::mutex mtx;
  std::condition_variable condition_var;
  bool ready = false;

  // Start the streamDepthMap function in a separate thread
  std::thread img_thread(streamDepthMap, std::ref(pipe),std::ref(cfg),
      std::ref(output_depth_map), std::ref(mtx), std::ref(condition_var), std::ref(ready));

  // Wait for the first available output_depth_map
  {
    std::unique_lock<std::mutex> lock(mtx);
    condition_var.wait(lock, [&ready] { return ready; });
  }
  // Copy the output_depth_map to depth_map so that we can use it in the rest of the program
  depth_map = output_depth_map.clone();

  // Convert the depth frame to a PCL point cloud
  pcl::PointCloud<pcl::PointXYZ>::Ptr pcl_cloud(new pcl::PointCloud<pcl::PointXYZ>), cloud_f(new pcl::PointCloud<pcl::PointXYZ>);
  for (int y = 0; y < depth_map.rows; ++y) {
    for (int x = 0; x < depth_map.cols; ++x) {
      if (uint16_t depth_value = depth_map.at<uint16_t>(y, x); depth_value > 0) {
        pcl::PointXYZ point;
        point.x = static_cast<float>(x);
        point.y = static_cast<float>(y);
        point.z = static_cast<float>(depth_value) * 0.001f; // Convert from mm to meters
        pcl_cloud->points.push_back(point);
      }
    }
  }
  pcl_cloud->width = static_cast<uint32_t>(pcl_cloud->points.size());
  pcl_cloud->height = 1;
  pcl_cloud->is_dense = false;
  pcl_cloud->sensor_origin_.setZero();

  std::cout << "PointCloud captured from Realsense camera has: " << pcl_cloud->size() << " data points." << std::endl;

  // Create the filtering object: downsample the dataset using a leaf size of 1cm
  pcl::VoxelGrid<pcl::PointXYZ> vg;
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_filtered (new pcl::PointCloud<pcl::PointXYZ>);
  vg.setInputCloud (pcl_cloud);
  vg.setLeafSize (0.01f, 0.01f, 0.01f);
  vg.filter (*cloud_filtered);
  std::cout << "PointCloud after filtering has: " << cloud_filtered->size ()  << " data points." << std::endl; //*

  // Create the segmentation object for the planar model and set all the parameters
  pcl::SACSegmentation<pcl::PointXYZ> seg;
  pcl::PointIndices::Ptr inliers (new pcl::PointIndices);
  pcl::ModelCoefficients::Ptr coefficients (new pcl::ModelCoefficients);
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_plane (new pcl::PointCloud<pcl::PointXYZ> ());
  pcl::PCDWriter writer;
  seg.setOptimizeCoefficients (true);
  seg.setModelType (pcl::SACMODEL_PLANE);
  seg.setMethodType (pcl::SAC_RANSAC);
  seg.setMaxIterations (100);
  seg.setDistanceThreshold (0.02);

  int nr_points = static_cast<int>(cloud_filtered->size());
  while (cloud_filtered->size () > 0.3 * nr_points)
  {
    // Segment the largest planar component from the remaining cloud
    seg.setInputCloud (cloud_filtered);
    seg.segment (*inliers, *coefficients);
    if (inliers->indices.empty())
    {
      std::cout << "Could not estimate a planar model for the given dataset." << std::endl;
      break;
    }

    // Extract the planar inliers from the input cloud
    pcl::ExtractIndices<pcl::PointXYZ> extract;
    extract.setInputCloud (cloud_filtered);
    extract.setIndices (inliers);
    extract.setNegative (false);

    // Get the points associated with the planar surface
    extract.filter (*cloud_plane);
    std::cout << "PointCloud representing the planar component: " << cloud_plane->size () << " data points." << std::endl;

    // Remove the planar inliers, extract the rest
    extract.setNegative (true);
    extract.filter (*cloud_f);
    *cloud_filtered = *cloud_f;
  }

  // Creating the KdTree object for the search method of the extraction
  pcl::search::KdTree<pcl::PointXYZ>::Ptr tree (new pcl::search::KdTree<pcl::PointXYZ>);
  tree->setInputCloud (cloud_filtered);

  std::vector<pcl::PointIndices> cluster_indices;
  pcl::EuclideanClusterExtraction<pcl::PointXYZ> ec;
  ec.setClusterTolerance (0.02); // 2cm
  ec.setMinClusterSize (100);
  ec.setMaxClusterSize (25000);
  ec.setSearchMethod (tree);
  ec.setInputCloud (cloud_filtered);
  ec.extract (cluster_indices);

  int j = 0;

  // If no clusters are found, output a message
  if (cluster_indices.empty())
  {
    std::cout << "No clusters found." << std::endl;
    return (0);
  }

  for (const auto& cluster : cluster_indices)
  {
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_cluster (new pcl::PointCloud<pcl::PointXYZ>);
    for (const auto& idx : cluster.indices) {
      cloud_cluster->push_back((*cloud_filtered)[idx]);
    } //*
    cloud_cluster->width = cloud_cluster->size ();
    cloud_cluster->height = 1;
    cloud_cluster->is_dense = true;

    std::cout << "PointCloud representing the Cluster: " << cloud_cluster->size () << " data points." << std::endl;
    std::stringstream ss;
    ss << std::setw(4) << std::setfill('0') << j;
    writer.write<pcl::PointXYZ> (output_folder + "cloud_cluster_" + ss.str () + ".pcd", *cloud_cluster, false);
    j++;
  }

  return (0);
}