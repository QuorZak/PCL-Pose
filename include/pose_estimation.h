#pragma once
// Umbrella header for the pose library: one include that gives the programs everything in include/pcl_pose/
// together with the third-party headers they use.
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

#include <flann/flann.h>
#include <flann/io/hdf5.h>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <cfloat>
#include <thread>
#include <mutex>
#include <utility>
#include <condition_variable>
#include <Eigen/Dense>
#include <cmath>
#include <chrono>

#include <opencv2/opencv.hpp>
#include <opencv2/core/eigen.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "pcl_pose/parameters.h"
#include "pcl_pose/config.h"
#include "pcl_pose/camera.h"
#include "pcl_pose/segmentation.h"
#include "pcl_pose/vfh.h"
#include "pcl_pose/files.h"
#include "pcl_pose/library.h"
#include "pcl_pose/viewer.h"
#include "pcl_pose/pose.h"
