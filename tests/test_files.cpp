#include <gtest/gtest.h>

#include <filesystem>
#include <fstream>

#include "pcl_pose/files.h"

using namespace pcl_pose;
namespace fs = std::filesystem;

namespace {

// Each test gets its own empty folder in the system's temporary folder, which is removed afterwards
class FilesTest : public ::testing::Test {
protected:
  void SetUp() override {
    const auto* info = ::testing::UnitTest::GetInstance()->current_test_info();
    folder_ = fs::temp_directory_path() / (std::string("pcl_pose_") + info->test_suite_name() + "_" + info->name());
    fs::remove_all(folder_);
    fs::create_directories(folder_);
  }
  void TearDown() override { fs::remove_all(folder_); }

  void touch(const std::string& name) const { std::ofstream(folder_ / name).put('x'); }
  [[nodiscard]] std::string path(const std::string& name) const { return (folder_ / name).string(); }

  fs::path folder_;
};

}  // namespace

TEST_F(FilesTest, ListsThePcdFilesWithThePrefixInOrder) {
  touch("a_0002.pcd");
  touch("a_0001.pcd");
  touch("a_0001.txt");
  touch("b_0001.pcd");

  const std::vector<std::string> files = listPcdFiles(folder_.string(), "a_");
  ASSERT_EQ(files.size(), 2u);
  EXPECT_EQ(files[0], path("a_0001.pcd"));
  EXPECT_EQ(files[1], path("a_0002.pcd"));
}

TEST_F(FilesTest, ListsEveryPcdFileWithoutAPrefix) {
  touch("a_0001.pcd");
  touch("b_0001.pcd");
  touch("notes.txt");
  EXPECT_EQ(listPcdFiles(folder_.string()).size(), 2u);
}

TEST_F(FilesTest, ListingAMissingFolderGivesNothing) {
  EXPECT_TRUE(listPcdFiles(path("does_not_exist"), "a_").empty());
}

TEST_F(FilesTest, TheNextNumberIsOneMoreThanTheHighest) {
  touch("obj_0001.pcd");
  touch("obj_0001_vfh.pcd");
  touch("obj_0003.pcd");
  touch("obj_0003_vfh.pcd");
  touch("other_0009.pcd"); // another object
  EXPECT_EQ(nextFileNumber(folder_.string(), "obj"), 4);
}

TEST_F(FilesTest, TheNextNumberIgnoresFilesThatDoNotFollowThePattern) {
  touch("obj_0002.pcd");
  touch("obj_abcd.pcd");
  touch("obj_12.pcd");
  touch("obj_0007.txt");
  touch("obj_.pcd");
  EXPECT_EQ(nextFileNumber(folder_.string(), "obj"), 3);
}

TEST_F(FilesTest, TheFirstNumberIsOne) {
  EXPECT_EQ(nextFileNumber(folder_.string(), "obj"), 1);
}

TEST_F(FilesTest, TheNextNumberCreatesAMissingFolder) {
  const std::string new_folder = path("not_yet_there");
  EXPECT_FALSE(fs::exists(new_folder));
  EXPECT_EQ(nextFileNumber(new_folder, "obj"), 1);
  EXPECT_TRUE(fs::is_directory(new_folder));
}
