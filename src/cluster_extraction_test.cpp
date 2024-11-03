#include <pcl/ModelCoefficients.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <iomanip> // for setw, setfill
#include <iostream>
#include <filesystem>
#include <regex>

#include <pose_estimation.h>

int main() {
  const std::string test_name = "spray_bottle_tall";
  const std::string output_folder = "../lab_data/test/";
  //const std::string output_folder = "../lab_data/" + test_name + "/";
  const float object_facing_angle = 315.0f; // Change this each capture
  const float calibration_angle_offset = -40.0f; // Set this if your camera is not quite aligned

  // Get the highest file number
  int file_num = 0;
  for (const auto& entry : std::filesystem::directory_iterator(output_folder)) {
    std::string filename = entry.path().filename().string();
    if (filename.find(test_name) != std::string::npos && filename.find(".pcd") != std::string::npos) {
      int num = std::stoi(filename.substr(test_name.size() + 1, 4));
      if (num > file_num) {
        file_num = num;
      }
    }
  }
  file_num += 1;

  // Declare any variables
  std::unique_ptr<Eigen::Matrix4f> global_rotation(new Eigen::Matrix4f());
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);

  rs2::pipeline pipe;
  rs2::config config;
  config.enable_stream(RS2_STREAM_DEPTH, cam_res_width, cam_res_height, RS2_FORMAT_Z16, cam_fps); // Use global parameters
  //config.enable_stream(RS2_STREAM_ACCEL, RS2_FORMAT_MOTION_XYZ32F);
  pipe.start(config);

  // Initialise the filters which will be applied to the depth frame
  initialise_filters();

  // Camera warmup - dropping several first frames to let auto-exposure stabilize
  for (int i = 0; i < 100; i++) {
    auto frames = pipe.wait_for_frames(); // purposely does nothing with the frames
  }

  rs2::frameset frames = pipe.wait_for_frames();
  rs2::frame depth = frames.get_depth_frame();
  depth = apply_depth_post_processing_filters(depth);

  /*// Get the camera orientation data
  rs2::motion_frame accel_frame = frames.first_or_default(RS2_STREAM_ACCEL);
  rs2_vector accel_data = accel_frame.get_motion_data();*/

  // Set the global origin and axes
  //setGlobalOriginAndAxes(*global_rotation, accel_data, object_facing_angle, calibration_angle_offset);

  cloud = depthFrameToPointCloud(depth, true, true);

  // apply the global rotation to the point cloud
  //transformPointCloud(*cloud, *cloud, *global_rotation);

  pipe.stop();

  // Filter and segment the point cloud
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_filtered(new pcl::PointCloud<pcl::PointXYZ>);
  std::vector<pcl::PointIndices> cluster_indices;
  filterAndSegmentPointCloud(cloud, cloud_filtered, cluster_indices, true);

  // Clear out all the files in the output folder
  const std::string clear_command = "rm -f " + output_folder + test_name + "_*.pcd";
  if (const int result = std::system(clear_command.c_str()); result != 0) {
    std::cerr << "Failed to execute rm command" << std::endl;
  } else {
    std::cout << "Cleared out all the files in the output folder" << std::endl;
  }

  if (cluster_indices.empty()) {
    std::cout << "No clusters found." << std::endl;
    return 1;
  }

  // Save the point clouds
  pcl::PCDWriter writer;
  std::string output_file;
  int j = 0;
  for (const auto& cluster : cluster_indices) {
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_cluster(new pcl::PointCloud<pcl::PointXYZ>);
    for (const auto& idx : cluster.indices) {
      cloud_cluster->push_back((*cloud_filtered)[idx]);
    }
    cloud_cluster->width = cloud_cluster->size();
    cloud_cluster->height = 1;
    cloud_cluster->is_dense = true;

    // Transform the point cloud using the new pose
    //transformPointCloud(*cloud_cluster, *cloud_cluster, *object_pose);

    std::cout << "PointCloud representing the Cluster: " << cloud_cluster->size() << " data points." << std::endl;
    std::stringstream ss;
    ss << std::setw(4) << std::setfill('0') << file_num++;
    output_file = output_folder + test_name + "_" + ss.str() + ".pcd";
    writer.write<pcl::PointXYZ>(output_folder + test_name + "_" + ss.str() + ".pcd", *cloud_cluster, false);

    // Get the VFH signature of the cluster then save
    pcl::PointCloud<pcl::VFHSignature308> signature;
    estimate_VFH(cloud_cluster, signature);
    std::string vfh_filename = output_folder + test_name + "_" + ss.str() + "_vfh.pcd";
    pcl::io::savePCDFile(vfh_filename, signature);

    j++;
  }

  const std::string files_to_show_pattern = output_folder + test_name + "_*.pcd";
  std::vector<std::string> files_to_show = globFiles(files_to_show_pattern);
  showPointClouds(files_to_show);

  //showPointClouds({output_file});

  return 0;
}