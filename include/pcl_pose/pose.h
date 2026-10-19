#pragma once
#include <cmath>

#include <Eigen/Dense>
#include <librealsense2/rs.hpp>
#include <opencv2/core.hpp>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#include "pcl_pose/parameters.h"

namespace pcl_pose {

// Keeps the current estimate of the pose, and a calibration adjuster that is set by measuring the error of the
// robot's position
class PoseManager {
public:
  PoseManager() : stored_pose(Eigen::Matrix4f::Identity()) {}

  void setPose(const Eigen::Matrix4f& pose) {
    stored_pose = pose;
  }

  [[nodiscard]] Eigen::Matrix4f getPose() const {
    return stored_pose;
  }

  void updatePose(const Eigen::Matrix4f& pose, const bool use_scaling_factor = false) {
    if (use_scaling_factor)
    {
      const float distance = (stored_pose.block<3, 1>(0, 3) - pose.block<3, 1>(0, 3)).norm();
      const float update_factor = calculateUpdateFactor(distance);
      // Blend the rotation with a spherical interpolation and the translation linearly. Blending the whole 4x4 matrix
      // linearly (as this used to) does not give a valid rotation.
      const Eigen::Matrix3f stored_rotation = stored_pose.block<3, 3>(0, 0);
      const Eigen::Matrix3f new_rotation = pose.block<3, 3>(0, 0);
      const Eigen::Quaternionf blended_rotation =
        Eigen::Quaternionf(stored_rotation).slerp(update_factor, Eigen::Quaternionf(new_rotation));
      Eigen::Matrix4f blended_pose = Eigen::Matrix4f::Identity();
      blended_pose.block<3, 3>(0, 0) = blended_rotation.toRotationMatrix();
      blended_pose.block<3, 1>(0, 3) =
        update_factor * pose.block<3, 1>(0, 3) + (1.0f - update_factor) * stored_pose.block<3, 1>(0, 3);
      stored_pose = blended_pose;
      return;
    }
    stored_pose = pose;
  }

  // get the pose calibration adjuster
  [[nodiscard]] Eigen::Matrix4f getPoseCalibrationAdjuster() const {
    return pose_calibration_adjuster;
  }
  // update the pose calibration adjuster
  void updatePoseCalibrationAdjuster(const Eigen::Matrix4f& pose) {
    pose_calibration_adjuster = pose;
  }
  void updatePoseCalibrationAdjusterXYZ(const float x = 0, const float y = 0, const float z = 0) {
    pose_calibration_adjuster(0, 3) += x;
    pose_calibration_adjuster(1, 3) += y;
    pose_calibration_adjuster(2, 3) += z;
  }

private:
  Eigen::Matrix4f stored_pose;
  Eigen::Matrix4f pose_calibration_adjuster = Eigen::Matrix4f::Identity();

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
cv::Point3f transformPoint(const Eigen::Matrix4f& pose, const cv::Point3f& point);

// Helper function to project a 3D point to 2D image point
cv::Point2f projectPoint(const cv::Point3f& point, const rs2_intrinsics& intrinsics);

// Function to estimate the pose of the object (ICP of the model cloud against the scene cloud)
Eigen::Matrix4f estimatePose(const pcl::PointCloud<pcl::PointXYZ>::Ptr& scene_cloud,
  const pcl::PointCloud<pcl::PointXYZ>::Ptr& model_cloud);

// Function to display the coordinate system on top of the video feed
void displayCoordinateSystem(const Eigen::Matrix4f& pose, cv::Mat& frame, const rs2_intrinsics& intrinsics);

// This function takes a point cloud and sets the origin to the center of the point cloud
// It also sets the axes to the principal axes of the point cloud, currently y = up, z = forward/front of the object
// 0 degrees is defined as directly facing the camera
// Used by cluster_extraction.cpp to place each library cloud in its own coordinate system.
void getPointCloudOriginAndAxes(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud_cluster, Eigen::Matrix4f& pose,
  float object_facing_angle, float calibration_angle_offset, const rs2_vector& accel_data);

// Get the angle (in degrees) that makes the y-axis line up with gravity, using the camera's accelerometer
// The cloud is only rotated if apply_rotation is set
void align_yaxis_with_gravity(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud_cluster,
  Eigen::Matrix4f& y_align_rotation, const rs2_vector& accel_data, float& angle,
  const bool& apply_rotation = false);

}  // namespace pcl_pose
