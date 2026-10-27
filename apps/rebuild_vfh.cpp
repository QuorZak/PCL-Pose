#include <pose_estimation.h>

using namespace pcl_pose;

// Checks the object library, and with --apply fixes it: every cloud is moved so that its centroid is at the origin
// (as in the research report) and the paired _vfh.pcd signature is recomputed from the cloud, so the pair always match.
// No camera is needed.
//
//   rebuild_vfh [-m <library folder>]            only lists each cloud and whether it is centred (changes nothing)
//   rebuild_vfh [-m <library folder>] --apply    centres the clouds and rewrites the clouds and the _vfh files
int main(int argc, char** argv) {
  std::string library_directory = "../lab_data/";
  pcl::console::parse_argument(argc, argv, "-m", library_directory);
  const bool apply = pcl::console::find_switch(argc, argv, "--apply");
  constexpr float centred_tolerance = 0.001f; // A cloud counts as centred if its centroid is within 1 mm of the origin

  const std::vector<std::string> clouds = findLibraryClouds(library_directory);
  if (clouds.empty()) {
    std::cerr << "No point clouds found in " << library_directory << std::endl;
    return 1;
  }

  int not_centred = 0;
  int rewritten = 0;
  pcl::PCDWriter writer;
  for (const auto& cloud_file : clouds) {
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
    if (pcl::io::loadPCDFile(cloud_file, *cloud) == -1) {
      std::cerr << "Couldn't read " << cloud_file << std::endl;
      continue;
    }

    const Eigen::Vector4f centroid = cloudCentroid(*cloud);
    const bool centred = centroid.head<3>().norm() <= centred_tolerance;
    std::cout << cloud_file << ": " << cloud->size() << " points, centroid (" << std::fixed << std::setprecision(4)
              << centroid[0] << ", " << centroid[1] << ", " << centroid[2] << ") m, "
              << (centred ? "centred" : "NOT centred") << std::endl;
    if (!centred) {
      not_centred++;
    }
    if (!apply) {
      continue;
    }

    if (!centred) {
      centreOnCentroid(*cloud);
      writer.write<pcl::PointXYZ>(cloud_file, *cloud, false);
    }
    // Always recompute the signature, so it is guaranteed to belong to the cloud that is saved next to it
    pcl::PointCloud<pcl::VFHSignature308> signature;
    estimate_VFH(cloud, signature);
    pcl::io::savePCDFile(vfhFileForCloud(cloud_file), signature);
    rewritten++;
  }

  std::cout << clouds.size() << " clouds, " << not_centred << " not centred." << std::endl;
  if (apply) {
    std::cout << "Rewrote the VFH signature of " << rewritten << " clouds." << std::endl;
  } else if (not_centred > 0) {
    std::cout << "Nothing was changed. Run again with --apply to centre them and recompute the signatures." << std::endl;
  }
  return 0;
}
