#include <gtest/gtest.h>

#include <cmath>

#include <pcl/common/centroid.h>

#include "pcl_pose/segmentation.h"

using namespace pcl_pose;

namespace {

// A flat table 0.6 m in front of the camera (z = 0.6), 0.6 m by 0.6 m, with a point every 3 mm
void addTable(pcl::PointCloud<pcl::PointXYZ>& scene) {
  for (int i = 0; i < 200; ++i) {
    for (int j = 0; j < 200; ++j) {
      scene.push_back(pcl::PointXYZ(-0.3f + 0.003f * static_cast<float>(i), -0.3f + 0.003f * static_cast<float>(j), 0.6f));
    }
  }
}

// An upright can standing on the table: a cylinder along z from the table (z = 0.6) towards the camera (z = 0.4)
void addCan(pcl::PointCloud<pcl::PointXYZ>& scene, const float centre_x, const float centre_y, const float radius) {
  for (int row = 0; row <= 100; ++row) {
    const float z = 0.6f - 0.002f * static_cast<float>(row);
    for (int step = 0; step < 120; ++step) {
      const float angle = 2.0f * static_cast<float>(M_PI) * static_cast<float>(step) / 120.0f;
      scene.push_back(pcl::PointXYZ(centre_x + radius * std::cos(angle), centre_y + radius * std::sin(angle), z));
    }
  }
}

}  // namespace

TEST(Segmentation, FindsOneObjectOnATable) {
  pcl::PointCloud<pcl::PointXYZ>::Ptr scene(new pcl::PointCloud<pcl::PointXYZ>);
  addTable(*scene);
  addCan(*scene, 0.05f, -0.02f, 0.04f);

  pcl::PointCloud<pcl::PointXYZ>::Ptr filtered(new pcl::PointCloud<pcl::PointXYZ>);
  std::vector<pcl::PointIndices> clusters;
  filterAndSegmentPointCloud(scene, filtered, clusters);

  // The table is removed as a plane, and the can is left as the one cluster
  ASSERT_EQ(clusters.size(), 1u);

  pcl::PointCloud<pcl::PointXYZ> can;
  for (const auto index : clusters[0].indices) {
    can.push_back((*filtered)[index]);
  }
  Eigen::Vector4f centroid;
  pcl::compute3DCentroid(can, centroid);
  EXPECT_NEAR(centroid[0], 0.05f, 0.01f);
  EXPECT_NEAR(centroid[1], -0.02f, 0.01f);
  EXPECT_GT(centroid[2], 0.4f);
  EXPECT_LT(centroid[2], 0.6f);
}

TEST(Segmentation, ASceneWithOnlyATableHasNoClusters) {
  pcl::PointCloud<pcl::PointXYZ>::Ptr scene(new pcl::PointCloud<pcl::PointXYZ>);
  addTable(*scene);

  pcl::PointCloud<pcl::PointXYZ>::Ptr filtered(new pcl::PointCloud<pcl::PointXYZ>);
  std::vector<pcl::PointIndices> clusters;
  filterAndSegmentPointCloud(scene, filtered, clusters);

  EXPECT_TRUE(clusters.empty());
}
