#include <gtest/gtest.h>

#include <cmath>

#include "pcl_pose/target.h"

using namespace pcl_pose;

namespace {

// A cylinder of points, 21 rings of 36 points. Its axis runs along y from y_bottom up by height, and it is centred on
// x = centre_x and z = centre_z, so the point closest to the camera (smallest z) is at z = centre_z - radius.
pcl::PointCloud<pcl::PointXYZ> makeCylinder(const float centre_x, const float y_bottom, const float centre_z,
                                            const float radius, const float height) {
  pcl::PointCloud<pcl::PointXYZ> cloud;
  for (int row = 0; row <= 20; ++row) {
    const float y = y_bottom + height * static_cast<float>(row) / 20.0f;
    for (int step = 0; step < 36; ++step) {
      const float angle = 2.0f * static_cast<float>(M_PI) * static_cast<float>(step) / 36.0f;
      cloud.push_back(pcl::PointXYZ(centre_x + radius * std::cos(angle), y, centre_z + radius * std::sin(angle)));
    }
  }
  return cloud;
}

}  // namespace

TEST(Target, FindsThePointClosestToTheCameraInTheBand) {
  const float radius = 0.03f;
  const float height = 0.2f;
  const pcl::PointCloud<pcl::PointXYZ> cylinder = makeCylinder(0.0f, 0.0f, 0.4f, radius, height);

  Eigen::Vector3f closest = Eigen::Vector3f::Zero();
  ASSERT_TRUE(findClosestPointInBand(cylinder, closest));

  EXPECT_NEAR(closest.z(), 0.4f - radius, 1e-4f);
  EXPECT_NEAR(closest.x(), 0.0f, 1e-4f);
  // The band is the lower part of the object, from 65% to 85% of the way up the cylinder with the default settings
  EXPECT_GE(closest.y(), 0.6f * height);
  EXPECT_LE(closest.y(), 0.9f * height);
}

TEST(Target, WorksWithNegativeCoordinates) {
  // The maxima used to start from numeric_limits::min() (the smallest positive number), which broke the band for
  // an object where every x or y is negative
  const float radius = 0.03f;
  const float height = 0.2f;
  const pcl::PointCloud<pcl::PointXYZ> cylinder = makeCylinder(-1.0f, -1.0f, 0.4f, radius, height);

  Eigen::Vector3f closest = Eigen::Vector3f::Zero();
  ASSERT_TRUE(findClosestPointInBand(cylinder, closest));

  EXPECT_NEAR(closest.z(), 0.4f - radius, 1e-4f);
  EXPECT_NEAR(closest.x(), -1.0f, 1e-4f);
  EXPECT_GE(closest.y(), -1.0f + 0.6f * height);
  EXPECT_LE(closest.y(), -1.0f + 0.9f * height);
}

TEST(Target, FillsInTheBandAroundTheTarget) {
  const float height = 0.2f;
  const pcl::PointCloud<pcl::PointXYZ> cylinder = makeCylinder(0.0f, 0.0f, 0.4f, 0.03f, height);

  Eigen::Vector3f closest = Eigen::Vector3f::Zero();
  pcl::PointCloud<pcl::PointXYZ> band;
  ASSERT_TRUE(findClosestPointInBand(cylinder, closest, &band));

  ASSERT_FALSE(band.empty());
  EXPECT_LT(band.size(), cylinder.size());
  EXPECT_EQ(band.width, band.size());
  EXPECT_EQ(band.height, 1u);
  for (const auto& point : band.points) {
    EXPECT_GE(point.y, 0.6f * height);
    EXPECT_LE(point.y, 0.9f * height);
    EXPECT_GE(point.z, closest.z());
  }
}

TEST(Target, TheBandSettingsMoveTheTarget) {
  const float height = 0.2f;
  const pcl::PointCloud<pcl::PointXYZ> cylinder = makeCylinder(0.0f, 0.0f, 0.4f, 0.03f, height);

  // A band in the upper half instead of the lower half (start and stop are negative: above the middle)
  TargetBand upper;
  upper.start = -0.7f;
  upper.stop = -0.3f;
  Eigen::Vector3f closest = Eigen::Vector3f::Zero();
  ASSERT_TRUE(findClosestPointInBand(cylinder, closest, nullptr, upper));
  EXPECT_LT(closest.y(), 0.5f * height);
}

TEST(Target, AnEmptyCloudHasNoTarget) {
  const pcl::PointCloud<pcl::PointXYZ> empty;
  Eigen::Vector3f closest = Eigen::Vector3f::Zero();
  pcl::PointCloud<pcl::PointXYZ> band;
  band.push_back(pcl::PointXYZ(1.0f, 2.0f, 3.0f));
  EXPECT_FALSE(findClosestPointInBand(empty, closest, &band));
  EXPECT_TRUE(band.empty());
}

TEST(Target, ASinglePointIsItsOwnTarget) {
  pcl::PointCloud<pcl::PointXYZ> cloud;
  cloud.push_back(pcl::PointXYZ(0.1f, 0.2f, 0.3f));
  Eigen::Vector3f closest = Eigen::Vector3f::Zero();
  ASSERT_TRUE(findClosestPointInBand(cloud, closest));
  EXPECT_TRUE(closest.isApprox(Eigen::Vector3f(0.1f, 0.2f, 0.3f)));
}
