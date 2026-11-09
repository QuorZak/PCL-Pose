#include <pose_estimation.h>

using namespace pcl_pose;

// Captures one depth frame from the RealSense, filters it the way the live programs do, and saves the whole scene
// (not only the clusters) as a point cloud, so that it can be replayed later without a camera using `replay`.
// The scene is saved as ../scenes/<name>_NNNN.pcd, and the folder is created if it does not exist.
//
// Options (all optional):
//   --config <file>   YAML file with the parameters (default: ../config/default.yaml if it exists)
//   --name <name>     the name the saved file starts with (default: scene)
int main(int argc, char** argv) {
  if (!loadConfigFromArguments(argc, argv)) {
    return 1;
  }
  std::string name = "scene";
  pcl::console::parse_argument(argc, argv, "--name", name);
  const std::string output_folder = "../scenes/";
  const int file_num = nextFileNumber(output_folder, name);

  rs2::pipeline pipe;
  rs2::config config;
  config.enable_stream(RS2_STREAM_DEPTH, cam_res_width, cam_res_height, RS2_FORMAT_Z16, cam_fps);
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
  const pcl::PointCloud<pcl::PointXYZ>::Ptr cloud = depthFrameToPointCloud(depth, true, true);
  pipe.stop();

  std::stringstream ss;
  ss << std::setw(4) << std::setfill('0') << file_num;
  const std::string output_file = output_folder + name + "_" + ss.str() + ".pcd";
  if (pcl::io::savePCDFileBinary(output_file, *cloud) != 0) {
    std::cerr << "Failed to save " << output_file << std::endl;
    return 1;
  }
  std::cout << "Saved " << output_file << " (" << cloud->size() << " points). Replay it with: ./replay " << output_file
            << std::endl;
  return 0;
}
