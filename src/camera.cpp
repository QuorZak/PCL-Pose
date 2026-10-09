#include "pcl_pose/camera.h"
#include "pcl_pose/parameters.h"

#include <iostream>
#include <stdexcept>

namespace {
// Post-processing filters for the depth frame
// Threshold filter, Temporal filter, Hole filling filter
rs2::threshold_filter threshold_filter;
rs2::temporal_filter temporal_filter;
rs2::hole_filling_filter hole_filling_filter;
}  // namespace

std::tuple<int, int, int, int> get_crop_points(const int width, const int height, const int percentage) {
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

void initialise_filters() {
  threshold_filter.set_option(RS2_OPTION_MIN_DISTANCE, depth_filter_min_distance);
  threshold_filter.set_option(RS2_OPTION_MAX_DISTANCE, depth_filter_max_distance);

  temporal_filter.set_option(RS2_OPTION_FILTER_SMOOTH_ALPHA, depth_filter_smooth_alpha);
  temporal_filter.set_option(RS2_OPTION_FILTER_SMOOTH_DELTA, depth_filter_smooth_delta);
  temporal_filter.set_option(RS2_OPTION_HOLES_FILL, depth_filter_temporal_holes_fill);

  hole_filling_filter.set_option(RS2_OPTION_HOLES_FILL, depth_filter_holes_fill);
}

rs2::depth_frame apply_depth_post_processing_filters(rs2::depth_frame depth) {
  depth = threshold_filter.process(depth);
  depth = temporal_filter.process(depth);
  //depth = hole_filling_filter.process(depth);
  return depth;
}

pcl::PointCloud<pcl::PointXYZ>::Ptr depthFrameToPointCloud(const rs2::depth_frame& depth, const bool crop,
  const bool debugging) {
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
