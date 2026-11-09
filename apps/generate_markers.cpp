#include <marker_pose_estimation.h>

// Generate and save OpenCV ArUco markers into the f_markers folder.
// Eight markers are created in total, starting at aruco_marker_id and adding 11 each time (11, 22, ... 88).
int main() {
  const cv::aruco::Dictionary dictionary = cv::aruco::getPredefinedDictionary(aruco_dict_number);
  cv::Mat marker_image;
  for (int marker_id = aruco_marker_id; marker_id <= 88; marker_id += 11) {
    cv::aruco::generateImageMarker(dictionary, marker_id, aruco_marker_pixels, marker_image, 1);
    const std::string file_name = "marker_" + std::to_string(marker_id)
      + "_dict_" + std::to_string(aruco_dict_number) + "_size_" + std::to_string(aruco_marker_pixels) + ".png";
    cv::imwrite(f_markers_location + file_name, marker_image);
  }
  return 0;
}
