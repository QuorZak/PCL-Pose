#include <gtest/gtest.h>

#include <stdexcept>

#include "pcl_pose/camera.h"

using namespace pcl_pose;

TEST(CropPoints, ReducesToTheCentreOfTheImage) {
  // 40% of a 640x480 image, centred
  const auto [x_start, x_stop, y_start, y_stop] = get_crop_points(640, 480, 40);
  EXPECT_EQ(x_start, 192);
  EXPECT_EQ(x_stop, 448);
  EXPECT_EQ(y_start, 144);
  EXPECT_EQ(y_stop, 336);
}

TEST(CropPoints, FullImageAt100Percent) {
  const auto [x_start, x_stop, y_start, y_stop] = get_crop_points(848, 480, 100);
  EXPECT_EQ(x_start, 0);
  EXPECT_EQ(x_stop, 848);
  EXPECT_EQ(y_start, 0);
  EXPECT_EQ(y_stop, 480);
}

TEST(CropPoints, NothingLeftAtZeroPercent) {
  const auto [x_start, x_stop, y_start, y_stop] = get_crop_points(640, 480, 0);
  EXPECT_EQ(x_start, x_stop);
  EXPECT_EQ(y_start, y_stop);
  EXPECT_EQ(x_start, 320);
  EXPECT_EQ(y_start, 240);
}

TEST(CropPoints, StaysInsideTheImage) {
  for (int percentage = 0; percentage <= 100; percentage += 5) {
    const auto [x_start, x_stop, y_start, y_stop] = get_crop_points(848, 480, percentage);
    EXPECT_GE(x_start, 0);
    EXPECT_LE(x_stop, 848);
    EXPECT_GE(y_start, 0);
    EXPECT_LE(y_stop, 480);
    EXPECT_LE(x_start, x_stop);
    EXPECT_LE(y_start, y_stop);
  }
}

TEST(CropPoints, RejectsAPercentageOutsideZeroToOneHundred) {
  EXPECT_THROW(get_crop_points(640, 480, -1), std::invalid_argument);
  EXPECT_THROW(get_crop_points(640, 480, 101), std::invalid_argument);
}
