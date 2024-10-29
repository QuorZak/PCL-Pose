#include <pcl/ModelCoefficients.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <iomanip> // for setw, setfill
#include <iostream>
#include <cstdio> // for std::remove
#include <filesystem>
#include <regex>

#include <pose_estimation.h>

int main () {
  const std::string object_name = "spray_bottle_tall";
  const std::string output_folder = "../lab_data/" + object_name + "/";
  // Define the angle (in degrees) that the object is facing
  const std::regex float_regex(R"(^\d{1,3}(\.\d{1,6})?$)"); // Regex to match a float. Angle is between 0 and 360 and 6dp
  float object_facing_angle = 0.0f; // Defines the angle of the object in degrees. 0 degrees is facing the camera
  constexpr float calibration_angle_offset = -40.0f; // Set this if your camera is not quite aligned
  constexpr int desired_cluster_size = 1;

  // Declare any variables so they don't have to be redefined each time
  std::unique_ptr<Eigen::Matrix4f> object_pose(new Eigen::Matrix4f());
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);

  // Get the highest file number
  int file_num = 0;
  for (const auto& entry : std::filesystem::directory_iterator(output_folder)) {
    std::string filename = entry.path().filename().string();
    if (filename.find(object_name) != std::string::npos && filename.find(".pcd") != std::string::npos) {
      int num = std::stoi(filename.substr(object_name.size() + 1, 4));
      if (num > file_num) {
        file_num = num;
      }
    }
  }
  file_num += 1;

  // Initialise the Realsense pipeline
  rs2::pipeline pipe;
  rs2::config config;
  config.enable_stream(RS2_STREAM_DEPTH, cam_res_width, cam_res_height, RS2_FORMAT_Z16, cam_fps); // Use global parameters
  config.enable_stream(RS2_STREAM_ACCEL, RS2_FORMAT_MOTION_XYZ32F);
  pipe.start(config);

  // Initialise the filters which will be applied to the depth frame
  initialise_filters();

  // Camera warmup - dropping several first frames to let auto-exposure stabilize
  for (int i = 0; i < 100; i++) {
    auto frames = pipe.wait_for_frames(); // purposely does nothing with the frames
  }
  std::cout << "Ready to capture point cloud. " << std::endl;

  // Get the camera orientation data
  rs2::frameset frames = pipe.wait_for_frames();
  rs2::motion_frame accel_frame = frames.first_or_default(RS2_STREAM_ACCEL);
  rs2_vector accel_data = accel_frame.get_motion_data();

  while (true) {
    std::cout << "Type in the angle of the object in degrees and press enter"<< std::endl;
    std::cout << "Or type Q to quit" << std::endl;

    std::string input;
    std::cin >> input;

    if (input == "Q" || input == "q") {
      break;
    }

    if (std::regex_match(input, float_regex)) {
      if (float angle = std::stof(input); angle >= 0.0f && angle <= 360.0f) {
        std::stringstream ss;
        ss << std::fixed << std::setprecision(4) << angle;
        object_facing_angle = std::stof(ss.str());
        std::cout << "Angle set to " << object_facing_angle << " degrees." << std::endl;
      } else {
        std::cout << "Invalid input. Please enter a number between 0 and 360." << std::endl;
        continue;
      }
    } else {
      std::cout << "Invalid input. Please enter a number between 0 and 360." << std::endl;
      continue;
    }

    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_filtered(new pcl::PointCloud<pcl::PointXYZ>);
    std::vector<pcl::PointIndices> cluster_indices;
    while (cluster_indices.size() != desired_cluster_size)
    {
      // Capture a frame
      frames = pipe.wait_for_frames();
      rs2::frame depth = frames.get_depth_frame();
      rs2::frame filtered_depth = apply_post_processing_filters(depth);

      // Convert the depth frame to a PCL point cloud
      cloud = depthFrameToPointCloud(filtered_depth, true, true);

      // Call the extracted function
      // temp variables to store the output
      pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_filtered_local(new pcl::PointCloud<pcl::PointXYZ>);
      std::vector<pcl::PointIndices> cluster_indices_local;
      filterAndSegmentPointCloud(cloud, cloud_filtered_local, cluster_indices_local, true);

      cloud_filtered = cloud_filtered_local;
      cluster_indices = cluster_indices_local;

      // Output a message but don't input a newline
      std::cout << "Clusters:" << cluster_indices.size() << "... ";
    }

    // If no clusters are found, output a message
    if (!cluster_indices.empty()) {
      int j = 0;
      pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_cluster(new pcl::PointCloud<pcl::PointXYZ>);
      std::vector<std::string> created_files;
      pcl::PCDWriter writer;
      for (const auto& cluster : cluster_indices) {
        cloud_cluster->clear(); // Ensure the cloud_cluster is cleared
        for (const auto& idx : cluster.indices) {
          cloud_cluster->push_back((*cloud_filtered)[idx]);
        }
        cloud_cluster->width = cloud_cluster->size();
        cloud_cluster->height = 1;
        cloud_cluster->is_dense = true;

        // Set the front direction of the point cloud
        getPointCloudOriginAndAxes(cloud_cluster,*object_pose, object_facing_angle, calibration_angle_offset, accel_data);

        // Transform the point cloud using the new pose
        transformPointCloud(*cloud_cluster, *cloud_cluster, *object_pose);

        // Save the point cloud
        std::cout << "PointCloud representing the Cluster: " << cloud_cluster->size() << " data points." << std::endl;
        std::stringstream ss;
        ss << std::setw(4) << std::setfill('0') << file_num++;
        std::string filename = output_folder + object_name + "_" + ss.str() + ".pcd";
        writer.write<pcl::PointXYZ>(filename, *cloud_cluster, false);
        created_files.push_back(filename);

        // Get the VFH signature of the cluster then save
        pcl::PointCloud<pcl::VFHSignature308> signature;
        estimate_VFH(cloud_cluster, signature);
        std::string vfh_filename = output_folder + object_name + "_" + ss.str() + "_vfh.pcd";
        pcl::io::savePCDFile(vfh_filename, signature);
        created_files.push_back(vfh_filename);

        j++;
      }

      // Display the results to review
      showPointClouds(created_files, true);

      std::cout << "Keep the result? y/[n]" << std::endl;
      char choice;
      std::cin >> choice;
      if (choice == 'y' || choice == 'Y') {
        std::cout << "Files saved." << std::endl;
        continue;
      } else {
        for (const auto& file : created_files) {
          std::remove(file.c_str());
        }
        std::cout << "Files deleted." << std::endl;
      }
    } else {
      std::cout << "No clusters found." << std::endl;
      std::cout << "Do you want to continue? y/[n]" << std::endl;
      char choice;
      std::cin >> choice;
      if (choice != 'y' && choice != 'Y') {
        break;
      }
    }
  }
  pipe.stop();
  return 0;
}