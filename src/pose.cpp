#include "pcl_pose/pose.h"
#include "pcl_pose/library.h"

#include <algorithm>
#include <cmath>

#include <opencv2/imgproc.hpp>

#include <pcl/common/centroid.h>
#include <pcl/common/transforms.h>
#include <pcl/registration/icp.h>

namespace pcl_pose {

cv::Point3f transformPoint(const Eigen::Matrix4f& pose, const cv::Point3f& point) {
  const Eigen::Vector4f point_homogeneous(point.x, point.y, point.z, 1.0);
  Eigen::Vector4f point_transformed = pose * point_homogeneous;
  return {point_transformed.x(), point_transformed.y(), point_transformed.z()};
}

cv::Point2f projectPoint(const cv::Point3f& point, const rs2_intrinsics& intrinsics) {
  float x_2d = intrinsics.fx * point.x / point.z + intrinsics.ppx;
  float y_2d = intrinsics.fy * point.y / point.z + intrinsics.ppy;
  return {x_2d, y_2d};
}

Eigen::Matrix4f estimatePose(const pcl::PointCloud<pcl::PointXYZ>::Ptr& scene_cloud,
  const pcl::PointCloud<pcl::PointXYZ>::Ptr& model_cloud) {

  pcl::IterativeClosestPoint<pcl::PointXYZ, pcl::PointXYZ> icp;
  icp.setInputSource(model_cloud);
  icp.setInputTarget(scene_cloud);
  icp.setMaximumIterations(icp_max_iterations);
  icp.setTransformationEpsilon(icp_transformation_epsilon);
  icp.setEuclideanFitnessEpsilon(icp_euclidean_fitness_epsilon);

  // Start from the model moved onto the scene cluster (centroid on centroid). The library clouds are centred, while
  // the scene cluster is around half a metre in front of the camera, which is too far for ICP to find from the identity.
  const Eigen::Vector4f scene_centroid = cloudCentroid(*scene_cloud);
  const Eigen::Vector4f model_centroid = cloudCentroid(*model_cloud);
  Eigen::Matrix4f initial_guess = Eigen::Matrix4f::Identity();
  initial_guess.block<3, 1>(0, 3) = scene_centroid.head<3>() - model_centroid.head<3>();

  pcl::PointCloud<pcl::PointXYZ> Final;
  icp.align(Final, initial_guess);

  if (icp.hasConverged()) {
    return icp.getFinalTransformation();
  } else {
    return Eigen::Matrix4f::Identity();
  }
}

void displayCoordinateSystem(const Eigen::Matrix4f& pose, cv::Mat& frame, const rs2_intrinsics& intrinsics) {
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

void getPointCloudOriginAndAxes(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud_cluster, Eigen::Matrix4f& pose,
  const float object_facing_angle, const float calibration_angle_offset, const rs2_vector& accel_data) {

  // Translate the point cloud to the origin (centroid). Its viewpoint (where the camera was) moves with it, so the VFH
  // signature of the centred cloud is still the one the camera sees.
  centreOnCentroid(*cloud_cluster);

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

void align_yaxis_with_gravity(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud_cluster,
  Eigen::Matrix4f& y_align_rotation, const rs2_vector& accel_data, float& angle,
  const bool& apply_rotation) {

  Eigen::Vector3f gravity_direction(accel_data.x, accel_data.y, accel_data.z);
  gravity_direction.normalize();

  // Calculate the angle between the y-axis and the gravity direction
  const Eigen::Vector3f y_axis = -Eigen::Vector3f::UnitY();
  const float unsigned_angle = std::acos(std::clamp(y_axis.dot(gravity_direction), -1.0f, 1.0f));
  const Eigen::Vector3f axis = y_axis.cross(gravity_direction);

  // The rotation that takes the gravity direction back onto the y-axis: the unsigned angle backwards about the axis
  // (the axis already carries the direction). Using the signed angle here, as this used to, rotated the wrong way
  // whenever the camera was tilted the other way.
  y_align_rotation.block<3, 3>(0, 0) = Eigen::Matrix3f::Identity();
  if (axis.norm() > 1e-6f) {
    y_align_rotation.block<3, 3>(0, 0) = Eigen::AngleAxisf(-unsigned_angle, axis.normalized()).toRotationMatrix();
  }

  // The angle that is reported (the RX for the operator) is signed: negative when the tilt is the other way about x
  angle = axis.dot(Eigen::Vector3f::UnitX()) < 0 ? -unsigned_angle : unsigned_angle;

  // Apply the of the resulting transformation to the point cloud to make it align with the world y-axis (gravity)
  if (apply_rotation) {
    transformPointCloud(*cloud_cluster, *cloud_cluster, y_align_rotation);
  }

  // make sure the angle comes out in degrees
  angle = 180.0f * angle / M_PI;
}

}  // namespace pcl_pose
