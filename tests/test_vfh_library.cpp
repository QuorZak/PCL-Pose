#include <gtest/gtest.h>

#include <string>

#include "pcl_pose/vfh_library.h"

using namespace pcl_pose;

namespace {

// The spray bottle library that is committed in the repository
const std::string spray_bottle_library = std::string(POSE_SOURCE_DIR) + "/lab_data/spray_bottle_tall";

vfh_model loadModel(const std::string& vfh_file) {
  vfh_model model;
  EXPECT_TRUE(load_vfh_histogram(vfh_file, model)) << "could not read " << vfh_file;
  return model;
}

}  // namespace

TEST(VfhLibrary, LoadsTheLibraryThatIsInTheRepository) {
  const VfhLibrary library(spray_bottle_library);
  EXPECT_FALSE(library.empty());
  EXPECT_EQ(library.size(), library.modelFiles().size());
  for (const auto& file : library.modelFiles()) {
    EXPECT_NE(file.find("_vfh.pcd"), std::string::npos) << file;
  }
}

TEST(VfhLibrary, AModelIsItsOwnClosestMatch) {
  const VfhLibrary library(spray_bottle_library);
  ASSERT_FALSE(library.empty());

  const std::size_t chosen = library.size() / 2;
  const vfh_model query = loadModel(library.modelFiles()[chosen]);
  const std::vector<VfhLibrary::Match> matches = library.nearest(query, 1);

  ASSERT_EQ(matches.size(), 1u);
  EXPECT_NEAR(matches[0].distance, 0.0f, 1e-4f);
}

TEST(VfhLibrary, TheMatchesComeBestFirstAndAreBoundedByK) {
  const VfhLibrary library(spray_bottle_library);
  ASSERT_GE(library.size(), 3u);

  const vfh_model query = loadModel(library.modelFiles()[0]);
  const std::vector<VfhLibrary::Match> matches = library.nearest(query, 3);

  ASSERT_EQ(matches.size(), 3u);
  EXPECT_LE(matches[0].distance, matches[1].distance);
  EXPECT_LE(matches[1].distance, matches[2].distance);
  for (const auto& match : matches) {
    ASSERT_LT(match.index, library.size());
    EXPECT_EQ(match.file, library.modelFiles()[match.index]);
  }
}

TEST(VfhLibrary, AskingForMoreThanTheLibraryHoldsGivesEverything) {
  const VfhLibrary library(spray_bottle_library);
  ASSERT_FALSE(library.empty());
  const vfh_model query = loadModel(library.modelFiles()[0]);
  EXPECT_EQ(library.nearest(query, 100000).size(), library.size());
}

TEST(VfhLibrary, NothingMatchesWhenKIsNotPositive) {
  const VfhLibrary library(spray_bottle_library);
  ASSERT_FALSE(library.empty());
  const vfh_model query = loadModel(library.modelFiles()[0]);
  EXPECT_TRUE(library.nearest(query, 0).empty());
  EXPECT_TRUE(library.nearest(query, -1).empty());
}

TEST(VfhLibrary, ASignatureWithTheWrongNumberOfBinsDoesNotMatch) {
  const VfhLibrary library(spray_bottle_library);
  ASSERT_FALSE(library.empty());
  vfh_model query;
  query.second.assign(10, 1.0f);
  EXPECT_TRUE(library.nearest(query, 3).empty());
}

TEST(VfhLibrary, AFolderThatDoesNotExistGivesAnEmptyLibrary) {
  const VfhLibrary library("this/folder/does/not/exist");
  EXPECT_TRUE(library.empty());
  EXPECT_EQ(library.size(), 0u);

  vfh_model query;
  query.second.assign(308, 1.0f);
  EXPECT_TRUE(library.nearest(query, 3).empty());
}
