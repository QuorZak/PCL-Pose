#include <gtest/gtest.h>

#include <cmath>

#include "pcl_pose/pose.h"

using namespace pcl_pose;

namespace {

// A pose that is a rotation about z by angle_z (radians) and a translation
Eigen::Matrix4f makePose(const float angle_z, const Eigen::Vector3f& translation) {
  Eigen::Matrix4f pose = Eigen::Matrix4f::Identity();
  pose.block<3, 3>(0, 0) = Eigen::AngleAxisf(angle_z, Eigen::Vector3f::UnitZ()).toRotationMatrix();
  pose.block<3, 1>(0, 3) = translation;
  return pose;
}

// The pose manager reads some of the global parameters, so put them back after each test
class PoseManagerTest : public ::testing::Test {
protected:
  void SetUp() override {
    saved_start_scale_distance_ = start_scale_distance;
    saved_stop_scale_distance_ = stop_scale_distance;
    saved_stability_factor_ = stability_factor;
  }
  void TearDown() override {
    start_scale_distance = saved_start_scale_distance_;
    stop_scale_distance = saved_stop_scale_distance_;
    stability_factor = saved_stability_factor_;
  }

private:
  float saved_start_scale_distance_ = 0;
  float saved_stop_scale_distance_ = 0;
  PoseUpdateStabilityFactor saved_stability_factor_ = PoseUpdateStabilityFactor::Resistant;
};

}  // namespace

TEST_F(PoseManagerTest, StartsAtTheIdentity) {
  const PoseManager manager;
  EXPECT_TRUE(manager.getPose().isApprox(Eigen::Matrix4f::Identity()));
  EXPECT_TRUE(manager.getPoseCalibrationAdjuster().isApprox(Eigen::Matrix4f::Identity()));
}

TEST_F(PoseManagerTest, SetPoseAndUpdateWithoutScalingReplaceThePose) {
  PoseManager manager;
  const Eigen::Matrix4f first = makePose(0.3f, Eigen::Vector3f(0.1f, 0.2f, 0.3f));
  manager.setPose(first);
  EXPECT_TRUE(manager.getPose().isApprox(first));

  const Eigen::Matrix4f second = makePose(-0.5f, Eigen::Vector3f(1.0f, 0.0f, -1.0f));
  manager.updatePose(second);
  EXPECT_TRUE(manager.getPose().isApprox(second));
}

TEST_F(PoseManagerTest, ASmallMoveIsTakenInFull) {
  // Closer than start_scale_distance (1 cm): the update factor is 1
  PoseManager manager;
  const Eigen::Matrix4f new_pose = makePose(0.3f, Eigen::Vector3f(0.005f, 0.0f, 0.0f));
  manager.updatePose(new_pose, true);
  EXPECT_LT((manager.getPose() - new_pose).norm(), 1e-4f);
}

TEST_F(PoseManagerTest, AJumpBeyondTheStopDistanceOnlyMovesALittle) {
  // Further than stop_scale_distance (0.5 m): the update factor is 0.05
  PoseManager manager;
  manager.updatePose(makePose(0.0f, Eigen::Vector3f(1.0f, 0.0f, 0.0f)), true);
  EXPECT_NEAR(manager.getPose()(0, 3), 0.05f, 1e-5f);
}

TEST_F(PoseManagerTest, TheBlendedRotationIsAValidRotation) {
  stability_factor = PoseUpdateStabilityFactor::Linear;
  PoseManager manager;
  const float target_angle = 1.2f;
  // 0.1 m is between the start and the stop distance, so the pose is blended
  manager.updatePose(makePose(target_angle, Eigen::Vector3f(0.1f, 0.0f, 0.0f)), true);

  const Eigen::Matrix3f rotation = manager.getPose().block<3, 3>(0, 0);
  EXPECT_LT((rotation.transpose() * rotation - Eigen::Matrix3f::Identity()).norm(), 1e-5f);
  EXPECT_NEAR(rotation.determinant(), 1.0f, 1e-5f);

  // It has turned towards the new pose, but not all the way
  const float angle = std::atan2(rotation(1, 0), rotation(0, 0));
  EXPECT_GT(angle, 0.0f);
  EXPECT_LT(angle, target_angle);
  // and the bottom row is still that of a transform
  EXPECT_TRUE(manager.getPose().row(3).isApprox(Eigen::RowVector4f(0.0f, 0.0f, 0.0f, 1.0f)));
}

TEST(GravityAlignment, TheRotationTakesGravityBackOntoTheYAxisForATiltEitherWay) {
  // The camera tilted by 20 degrees about x, one way and then the other. Gravity is then measured 20 degrees away from
  // the camera's -y axis (the accelerometer reports the reaction to gravity, along -y when the camera is level).
  const float tilt = 20.0f * static_cast<float>(M_PI) / 180.0f;
  float angles[2] = {0.0f, 0.0f};
  for (int side = 0; side < 2; ++side) {
    const float sign = side == 0 ? 1.0f : -1.0f;
    const rs2_vector accel{0.0f, -9.81f * std::cos(tilt), sign * 9.81f * std::sin(tilt)};
    const pcl::PointCloud<pcl::PointXYZ>::Ptr unused(new pcl::PointCloud<pcl::PointXYZ>);
    Eigen::Matrix4f rotation = Eigen::Matrix4f::Identity();
    align_yaxis_with_gravity(unused, rotation, accel, angles[side]);

    const Eigen::Vector3f gravity = Eigen::Vector3f(accel.x, accel.y, accel.z).normalized();
    const Eigen::Vector3f aligned = rotation.block<3, 3>(0, 0) * gravity;
    EXPECT_TRUE(aligned.isApprox(-Eigen::Vector3f::UnitY(), 1e-4f)) << "side " << side << ": " << aligned.transpose();
    EXPECT_NEAR(std::fabs(angles[side]), 20.0f, 1e-3f);
  }
  // The reported angle tells the two tilts apart
  EXPECT_NEAR(angles[0], -angles[1], 1e-3f);
}

TEST_F(PoseManagerTest, CalibrationOffsetsAddUp) {
  PoseManager manager;
  manager.updatePoseCalibrationAdjusterXYZ(0.001f, 0.002f, 0.003f);
  manager.updatePoseCalibrationAdjusterXYZ(0.001f, 0.002f, 0.003f);
  const Eigen::Matrix4f adjuster = manager.getPoseCalibrationAdjuster();
  EXPECT_NEAR(adjuster(0, 3), 0.002f, 1e-6f);
  EXPECT_NEAR(adjuster(1, 3), 0.004f, 1e-6f);
  EXPECT_NEAR(adjuster(2, 3), 0.006f, 1e-6f);
  // the rotation part is untouched
  // (extra brackets: the comma in block<3, 3> would otherwise split the macro's argument)
  EXPECT_TRUE((adjuster.block<3, 3>(0, 0).isApprox(Eigen::Matrix3f::Identity())));
}
