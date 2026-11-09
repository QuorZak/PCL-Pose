#include <pose_estimation.h>

#include <pcl/filters/filter.h>

using namespace pcl_pose;

// Runs the whole recognition pipeline on a saved scene instead of the camera, so nothing needs to be plugged in:
// segment the scene into clusters, match each cluster's VFH signature against the object library, and work out the
// point to aim for with the best match. The scene is a point cloud as the camera saw it (use capture_scene to save one).
//
//   replay <scene.pcd> [options]
//
// Options (all optional):
//   --config <file>   YAML file with the parameters (default: ../config/default.yaml if it exists)
//   -m <folder>       folder with the reference library (default: model_directory in the config)
//   -k <number>       how many of the closest library models to list for each cluster (default: 3)
//   --expect <text>   exit with an error unless the best match has this text in its file name, for scripts and tests
//   --debug           print more about the cluster filtering
//
// Exit status: 0 when a best match was found (and it is the expected one), 1 for a usage or file error, 2 when no
// clusters were found in the scene, 3 when the best match was not the expected one.
int main(int argc, char** argv) {
  if (!loadConfigFromArguments(argc, argv)) {
    return 1;
  }

  const std::vector<int> scene_arguments = pcl::console::parse_file_extension_argument(argc, argv, ".pcd");
  if (scene_arguments.empty()) {
    std::cerr << "Usage: replay <scene.pcd> [--config <file>] [-m <library folder>] [-k <number>] "
                 "[--expect <text>] [--debug]" << std::endl;
    return 1;
  }
  const std::string scene_file = argv[scene_arguments[0]];

  int k = 3;
  std::string expected;
  pcl::console::parse_argument(argc, argv, "-m", model_directory);
  pcl::console::parse_argument(argc, argv, "-k", k);
  pcl::console::parse_argument(argc, argv, "--expect", expected);
  const bool debug = pcl::console::find_switch(argc, argv, "--debug");

  // The library of known objects
  const VfhLibrary library(model_directory);
  if (library.empty()) {
    std::cerr << "No VFH models found in " << model_directory << std::endl;
    return 1;
  }
  std::cout << "Library: " << library.size() << " models in " << model_directory << std::endl;

  // The scene
  pcl::PointCloud<pcl::PointXYZ>::Ptr scene(new pcl::PointCloud<pcl::PointXYZ>);
  if (pcl::io::loadPCDFile(scene_file, *scene) == -1) {
    std::cerr << "Couldn't read " << scene_file << std::endl;
    return 1;
  }
  std::vector<int> kept_indices;
  pcl::removeNaNFromPointCloud(*scene, *scene, kept_indices);
  std::cout << "Scene: " << scene_file << ", " << scene->size() << " points" << std::endl;

  // The same steps as the live programs: down-sample, remove the planes, and extract the clusters
  pcl::PointCloud<pcl::PointXYZ>::Ptr scene_filtered(new pcl::PointCloud<pcl::PointXYZ>);
  std::vector<pcl::PointIndices> cluster_indices;
  filterAndSegmentPointCloud(scene, scene_filtered, cluster_indices, debug);
  if (cluster_indices.empty()) {
    std::cout << "No clusters found." << std::endl;
    return 2;
  }
  std::cout << cluster_indices.size() << " cluster(s) found" << std::endl;

  // Match each cluster against the library, and keep the best match of all
  pcl::PointCloud<pcl::PointXYZ>::Ptr best_cluster(new pcl::PointCloud<pcl::PointXYZ>);
  VfhLibrary::Match best_match{0, std::numeric_limits<float>::max(), ""};
  int cluster_number = 0;
  for (const auto& cluster : cluster_indices) {
    cluster_number++;
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_cluster(new pcl::PointCloud<pcl::PointXYZ>);
    for (const auto& idx : cluster.indices) {
      cloud_cluster->push_back(scene_filtered->points[idx]);
    }
    pcl::PointCloud<pcl::VFHSignature308> signature;
    estimate_VFH(cloud_cluster, signature);
    vfh_model query_model;
    query_model.second.assign(signature.points[0].histogram, signature.points[0].histogram + 308);

    const std::vector<VfhLibrary::Match> matches = library.nearest(query_model, k);
    std::cout << "Cluster " << cluster_number << ": " << cloud_cluster->size() << " points" << std::endl;
    for (std::size_t i = 0; i < matches.size(); ++i) {
      std::cout << "  " << i + 1 << ". " << cloudFileForVfh(matches[i].file) << "  distance " << std::fixed
                << std::setprecision(1) << matches[i].distance << std::endl;
    }
    if (!matches.empty() && matches[0].distance < best_match.distance) {
      best_match = matches[0];
      *best_cluster = *cloud_cluster;
    }
  }
  if (best_cluster->empty()) {
    std::cout << "No match found." << std::endl;
    return 2;
  }
  const std::string best_file = cloudFileForVfh(best_match.file);
  std::cout << "Best match: " << best_file << " (distance " << std::fixed << std::setprecision(1)
            << best_match.distance << ")" << std::endl;

  // Where to aim, as manual_measure works it out (but without the accelerometer, so there is no RX)
  pcl::PointCloud<pcl::PointXYZ>::Ptr object_wrt_TCP(new pcl::PointCloud<pcl::PointXYZ>);
  transformPointCloud(*best_cluster, *object_wrt_TCP, camera_to_TCP);
  Eigen::Vector3f closest_point = Eigen::Vector3f::Zero();
  if (findClosestPointInBand(*object_wrt_TCP, closest_point)) {
    const Eigen::Vector3f target_mm = closest_point * 1000.0f;
    std::cout << "Closest point in the target band, in the TCP frame: x " << std::fixed << std::setprecision(2)
              << target_mm.x() << " mm, y " << target_mm.y() << " mm, z " << target_mm.z() << " mm" << std::endl;
  } else {
    std::cout << "No points in the target band." << std::endl;
  }

  if (!expected.empty() && best_file.find(expected) == std::string::npos) {
    std::cerr << "Expected the best match to contain '" << expected << "'" << std::endl;
    return 3;
  }
  return 0;
}
