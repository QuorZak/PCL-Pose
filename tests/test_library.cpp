#include <gtest/gtest.h>

#include <filesystem>
#include <fstream>

#include "pcl_pose/library.h"

using namespace pcl_pose;
namespace fs = std::filesystem;

namespace {

pcl::PointCloud<pcl::PointXYZ> makeCloud() {
  pcl::PointCloud<pcl::PointXYZ> cloud;
  cloud.push_back(pcl::PointXYZ(1.0f, 2.0f, 3.0f));
  cloud.push_back(pcl::PointXYZ(3.0f, 2.0f, 1.0f));
  cloud.push_back(pcl::PointXYZ(2.0f, 5.0f, 5.0f));
  return cloud; // The centroid is (2, 3, 3)
}

}  // namespace

TEST(Library, FileNamesGoBackAndForthBetweenACloudAndItsSignature) {
  EXPECT_EQ(vfhFileForCloud("lab_data/bottle/bottle_0001.pcd"), "lab_data/bottle/bottle_0001_vfh.pcd");
  EXPECT_EQ(cloudFileForVfh("lab_data/bottle/bottle_0001_vfh.pcd"), "lab_data/bottle/bottle_0001.pcd");
  EXPECT_EQ(cloudFileForVfh(vfhFileForCloud("a/b_0042.pcd")), "a/b_0042.pcd");
}

TEST(Library, ANameThatIsNotASignatureFileIsLeftAlone) {
  EXPECT_EQ(cloudFileForVfh("bottle_0001.pcd"), "bottle_0001.pcd");
  EXPECT_EQ(cloudFileForVfh("vfh"), "vfh");
  EXPECT_EQ(cloudFileForVfh(""), "");
}

TEST(Library, TheCentroidIsTheAverageOfThePoints) {
  const Eigen::Vector4f centroid = cloudCentroid(makeCloud());
  EXPECT_NEAR(centroid[0], 2.0f, 1e-5f);
  EXPECT_NEAR(centroid[1], 3.0f, 1e-5f);
  EXPECT_NEAR(centroid[2], 3.0f, 1e-5f);
}

TEST(Library, CentringMovesTheCentroidToTheOrigin) {
  pcl::PointCloud<pcl::PointXYZ> cloud = makeCloud();
  const Eigen::Vector4f moved_from = centreOnCentroid(cloud);

  EXPECT_NEAR(moved_from[0], 2.0f, 1e-5f);
  EXPECT_NEAR(moved_from[1], 3.0f, 1e-5f);
  EXPECT_NEAR(moved_from[2], 3.0f, 1e-5f);

  const Eigen::Vector4f centroid = cloudCentroid(cloud);
  EXPECT_NEAR(centroid[0], 0.0f, 1e-5f);
  EXPECT_NEAR(centroid[1], 0.0f, 1e-5f);
  EXPECT_NEAR(centroid[2], 0.0f, 1e-5f);

  // The shape is kept: the first point was (-1, -1, 0) from the centroid
  EXPECT_NEAR(cloud[0].x, -1.0f, 1e-5f);
  EXPECT_NEAR(cloud[0].y, -1.0f, 1e-5f);
  EXPECT_NEAR(cloud[0].z, 0.0f, 1e-5f);

  // and the viewpoint (the camera, at the origin before) has moved with the cloud
  EXPECT_NEAR(cloud.sensor_origin_[0], -2.0f, 1e-5f);
  EXPECT_NEAR(cloud.sensor_origin_[1], -3.0f, 1e-5f);
  EXPECT_NEAR(cloud.sensor_origin_[2], -3.0f, 1e-5f);
}

TEST(Library, TransformingMovesTheViewpointWithTheCloud) {
  pcl::PointCloud<pcl::PointXYZ> cloud = makeCloud();
  cloud.sensor_origin_ = Eigen::Vector4f(0.0f, 0.0f, 0.0f, 0.0f);
  // A quarter turn about z, then a move along x
  Eigen::Matrix4f transform = Eigen::Matrix4f::Identity();
  transform.block<3, 3>(0, 0) = Eigen::AngleAxisf(static_cast<float>(M_PI) / 2.0f, Eigen::Vector3f::UnitZ()).toRotationMatrix();
  transform(0, 3) = 10.0f;
  transformCloudAndViewpoint(cloud, transform);

  // (1, 2, 3) turns to (-2, 1, 3), then moves to (8, 1, 3)
  EXPECT_NEAR(cloud[0].x, 8.0f, 1e-5f);
  EXPECT_NEAR(cloud[0].y, 1.0f, 1e-5f);
  EXPECT_NEAR(cloud[0].z, 3.0f, 1e-5f);
  // The viewpoint at the origin only moves
  EXPECT_NEAR(cloud.sensor_origin_[0], 10.0f, 1e-5f);
  EXPECT_NEAR(cloud.sensor_origin_[1], 0.0f, 1e-5f);
  EXPECT_NEAR(cloud.sensor_origin_[2], 0.0f, 1e-5f);
}

TEST(Library, AViewpointInsideTheObjectIsLost) {
  pcl::PointCloud<pcl::PointXYZ> cloud = makeCloud(); // centroid (2, 3, 3), viewpoint at the origin
  EXPECT_FALSE(viewpointIsLost(cloud));
  cloud.sensor_origin_ = Eigen::Vector4f(2.0f, 3.0f, 3.01f, 0.0f);
  EXPECT_TRUE(viewpointIsLost(cloud));
}

TEST(Library, FindsTheCloudsButNotTheirSignaturesInAllTheSubFolders) {
  const fs::path root = fs::temp_directory_path() / "pcl_pose_library_find";
  fs::remove_all(root);
  fs::create_directories(root / "bottle");
  fs::create_directories(root / "mustard");
  for (const char* name : {"bottle/bottle_0001.pcd", "bottle/bottle_0001_vfh.pcd", "mustard/mustard_0001.pcd",
                           "mustard/mustard_0001_vfh.pcd", "notes.txt"}) {
    std::ofstream(root / name).put('x');
  }

  const std::vector<std::string> clouds = findLibraryClouds(root.string());
  ASSERT_EQ(clouds.size(), 2u);
  EXPECT_EQ(clouds[0], (root / "bottle" / "bottle_0001.pcd").string());
  EXPECT_EQ(clouds[1], (root / "mustard" / "mustard_0001.pcd").string());

  fs::remove_all(root);
}

TEST(Library, ALibraryFolderThatDoesNotExistHasNoClouds) {
  EXPECT_TRUE(findLibraryClouds("this/folder/does/not/exist").empty());
}
