#include <pose_estimation.h>

using namespace pcl_pose;

// Checks the object library, and with --apply tidies it: every usable cloud is moved so that its centroid is at the
// origin (as in the research report), together with its viewpoint (where the camera was, the VIEWPOINT in the .pcd
// file), and the paired _vfh.pcd signature is recomputed from the cloud, so the pair always match.
// Moving the viewpoint with the cloud keeps the signature the same as the one the camera sees. A cloud whose viewpoint
// was lost (it was centred without moving the viewpoint, as the old `extraction` did) can not be repaired: it is
// reported and left alone, and the object has to be captured again.
// No camera is needed.
//
//   rebuild_vfh [-m <library folder>]            only lists each cloud and its state (changes nothing)
//   rebuild_vfh [-m <library folder>] --apply    centres the usable clouds and rewrites them and their _vfh files
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
  int lost_viewpoint = 0;
  int can_centre = 0; // Not centred, but with a usable viewpoint
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
    const bool lost = viewpointIsLost(*cloud);
    std::cout << cloud_file << ": " << cloud->size() << " points, centroid (" << std::fixed << std::setprecision(4)
              << centroid[0] << ", " << centroid[1] << ", " << centroid[2] << ") m, viewpoint ("
              << cloud->sensor_origin_[0] << ", " << cloud->sensor_origin_[1] << ", " << cloud->sensor_origin_[2]
              << ") m, " << (centred ? "centred" : "not centred")
              << (lost ? ", VIEWPOINT LOST (capture it again)" : "") << std::endl;
    if (!centred) {
      not_centred++;
    }
    if (lost) {
      lost_viewpoint++;
      continue;
    }
    if (!centred) {
      can_centre++;
    }
    if (!apply) {
      continue;
    }

    if (!centred) {
      centreOnCentroid(*cloud); // moves the viewpoint too
      writer.write<pcl::PointXYZ>(cloud_file, *cloud, false);
    }
    // Always recompute the signature (with the cloud's viewpoint), so it is guaranteed to belong to the cloud that is
    // saved next to it
    pcl::PointCloud<pcl::VFHSignature308> signature;
    estimate_VFH(cloud, signature);
    pcl::io::savePCDFile(vfhFileForCloud(cloud_file), signature);
    rewritten++;
  }

  std::cout << clouds.size() << " clouds, " << not_centred << " not centred, " << lost_viewpoint
            << " with a lost viewpoint." << std::endl;
  if (apply) {
    std::cout << "Rewrote the VFH signature of " << rewritten << " clouds." << std::endl;
  } else if (can_centre > 0) {
    std::cout << "Nothing was changed. Run again with --apply to centre the usable clouds and recompute their "
                 "signatures." << std::endl;
  }
  if (lost_viewpoint > 0) {
    std::cout << "The clouds with a lost viewpoint can not be repaired: capture those objects again with extraction."
              << std::endl;
  }
  return 0;
}
