#pragma once
#include <tuple>

#include <librealsense2/rs.hpp>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

// This function takes the width and height of a depth image and returns the x and y start and stop points for cropping
// The start and stop points are the absolute vales of the start position and stop position of the crop
// E.g. start_x = 150, stop_x = 450, start_y = 100, stop_y = 350 for a 640x480 image
// The percentage is the value of what the image should be reduced to (i.e. 40% of the original image)
// Returns: x_start, x_stop, y_start, y_stop
std::tuple<int, int, int, int> get_crop_points(int width, int height, int percentage);

// Initialise the filters for the depth frame (threshold, temporal and hole filling)
void initialise_filters();

// Apply post-processing filters to the depth frame
rs2::depth_frame apply_depth_post_processing_filters(rs2::depth_frame depth);

// Convert the depth frame to a PCL point cloud
// Allows for optional cropping of the image if crop is set to true
// The crop is centered around the center of the image
pcl::PointCloud<pcl::PointXYZ>::Ptr depthFrameToPointCloud(const rs2::depth_frame& depth, bool crop = false,
  bool debugging = false);
