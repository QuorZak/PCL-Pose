#include <gtest/gtest.h>

#include <cmath>

#include <pcl/common/transforms.h>

#include "pcl_pose/library.h"
#include "pcl_pose/vfh.h"

using namespace pcl_pose;

namespace {

// The front half of an upright can as the camera sees it: a half cylinder along y, 0.4 m in front of the camera (the
// camera is at the origin looking along +z), facing the camera
pcl::PointCloud<pcl::PointXYZ>::Ptr makeCameraView() {
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
  const float radius = 0.04f;
  const float centre_z = 0.4f;
  for (int row = 0; row <= 50; ++row) {
    const float y = -0.1f + 0.004f * static_cast<float>(row);
    for (int step = 0; step <= 60; ++step) {
      // From the left edge, round the side facing the camera (smallest z), to the right edge
      const float angle = static_cast<float>(M_PI) * (1.0f + static_cast<float>(step) / 60.0f);
      cloud->push_back(pcl::PointXYZ(radius * std::cos(angle), y, centre_z + radius * std::sin(angle)));
    }
  }
  return cloud;
}

std::vector<float> signatureOf(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud) {
  pcl::PointCloud<pcl::VFHSignature308> signature;
  estimate_VFH(cloud, signature);
  EXPECT_EQ(signature.size(), 1u);
  return std::vector<float>(signature[0].histogram, signature[0].histogram + 308);
}

float totalDifference(const std::vector<float>& a, const std::vector<float>& b) {
  float total = 0.0f;
  for (std::size_t i = 0; i < a.size(); ++i) {
    total += std::fabs(a[i] - b[i]);
  }
  return total;
}

// A rotation of 30 degrees about y and 20 degrees about x, and a move of about 0.4 m
Eigen::Matrix4f someRigidTransform() {
  Eigen::Matrix4f transform = Eigen::Matrix4f::Identity();
  transform.block<3, 3>(0, 0) = (Eigen::AngleAxisf(0.52f, Eigen::Vector3f::UnitY())
                                 * Eigen::AngleAxisf(0.35f, Eigen::Vector3f::UnitX())).toRotationMatrix();
  transform.block<3, 1>(0, 3) = Eigen::Vector3f(0.05f, -0.1f, -0.4f);
  return transform;
}

}  // namespace

// Each sub-histogram of a signature adds up to 100, so the whole signature adds up to about 500. The tolerances below
// are a small fraction of that.

TEST(Vfh, MovingTheCloudWithItsViewpointKeepsTheSignature) {
  const auto camera_view = makeCameraView();
  const std::vector<float> seen_by_camera = signatureOf(camera_view);

  pcl::PointCloud<pcl::PointXYZ>::Ptr moved(new pcl::PointCloud<pcl::PointXYZ>(*camera_view));
  transformCloudAndViewpoint(*moved, someRigidTransform());
  EXPECT_LT(totalDifference(seen_by_camera, signatureOf(moved)), 2.0f);
}

TEST(Vfh, CentringWithTheViewpointKeepsTheSignature) {
  // What a library cloud saved by extraction now looks like
  const auto camera_view = makeCameraView();
  const std::vector<float> seen_by_camera = signatureOf(camera_view);

  pcl::PointCloud<pcl::PointXYZ>::Ptr centred(new pcl::PointCloud<pcl::PointXYZ>(*camera_view));
  centreOnCentroid(*centred);
  EXPECT_FALSE(viewpointIsLost(*centred));
  EXPECT_LT(totalDifference(seen_by_camera, signatureOf(centred)), 2.0f);
}

TEST(Vfh, CentringWithoutTheViewpointChangesTheSignature) {
  // What the old extraction did: the cloud was centred but its viewpoint stayed at the origin, so the camera seemed to
  // be inside the object. That signature does not match what the camera sees, which is why the old library is retired.
  const auto camera_view = makeCameraView();
  const std::vector<float> seen_by_camera = signatureOf(camera_view);

  pcl::PointCloud<pcl::PointXYZ>::Ptr centred(new pcl::PointCloud<pcl::PointXYZ>);
  const Eigen::Vector4f centroid = cloudCentroid(*camera_view);
  Eigen::Matrix4f to_origin = Eigen::Matrix4f::Identity();
  to_origin.block<3, 1>(0, 3) = -centroid.head<3>();
  pcl::transformPointCloud(*camera_view, *centred, to_origin); // the viewpoint is copied unchanged
  EXPECT_TRUE(viewpointIsLost(*centred));
  EXPECT_GT(totalDifference(seen_by_camera, signatureOf(centred)), 10.0f);
}
