#include <pose_estimation.h>
#include <cstdlib>
#include <thread>
#include <opencv2/objdetect/aruco_detector.hpp>

// Generate and save openCv aruco markers
void generateMarker() {
  const cv::aruco::Dictionary dictionary = cv::aruco::getPredefinedDictionary(aruco_dict_number);
  cv::Mat markerImage;
  // Start with aruco_marker_id, then do it 7 more times (8 total), add 11 each time
  for (int marker_id = aruco_marker_id; marker_id <= 88; marker_id = marker_id + 11) {
    generateImageMarker(dictionary, marker_id, aruco_marker_pixels, markerImage, 1);
    std::string f_marker_name = "marker_" + std::to_string(marker_id)
      + "_dict_" + std::to_string(aruco_dict_number) + "_size_" + std::to_string(aruco_marker_pixels) + ".png";
    const std::string filename = f_markers_location + f_marker_name;
    imwrite(filename, markerImage);
  }
}

// function to start up camera rgb, find specifically the marker generated in generate marker, and display it
void findMarkerAndPose() {
using namespace cv;
// Initialize the RealSense pipeline
rs2::pipeline pipe;
rs2::config config;
config.enable_stream(RS2_STREAM_COLOR, cam_res_width, cam_res_height, RS2_FORMAT_BGR8, cam_fps); // Use global parameters
pipe.start(config);

// Get the intrinsics of the color stream
rs2::pipeline_profile profile = pipe.get_active_profile();
auto color_stream = profile.get_stream(RS2_STREAM_COLOR).as<rs2::video_stream_profile>();
rs2_intrinsics intrinsics = color_stream.get_intrinsics();

// Camera matrix and distortion coefficients
cv::Mat camMatrix = (cv::Mat_<double>(3, 3) << intrinsics.fx, 0, intrinsics.ppx, 0, intrinsics.fy, intrinsics.ppy, 0, 0, 1);
cv::Mat distCoeffs = (cv::Mat_<double>(5, 1) << intrinsics.coeffs[0], intrinsics.coeffs[1], intrinsics.coeffs[2], intrinsics.coeffs[3], intrinsics.coeffs[4]);

  // set coordinate system
  float markerLength = 0.1f;
  cv::Mat objPoints(4, 1, CV_32FC3);
  objPoints.ptr<Vec3f>(0)[0] = Vec3f(-markerLength/2.f, markerLength/2.f, 0);
  objPoints.ptr<Vec3f>(0)[1] = Vec3f(markerLength/2.f, markerLength/2.f, 0);
  objPoints.ptr<Vec3f>(0)[2] = Vec3f(markerLength/2.f, -markerLength/2.f, 0);
  objPoints.ptr<Vec3f>(0)[3] = Vec3f(-markerLength/2.f, -markerLength/2.f, 0);

  // Create the detector
  cv::aruco::DetectorParameters detectorParams = cv::aruco::DetectorParameters();
  cv::aruco::Dictionary dictionary = aruco::getPredefinedDictionary(aruco_dict_number);
  cv::aruco::ArucoDetector detector(dictionary, detectorParams);

while (true) {
  // Get the frames
  auto frames = pipe.wait_for_frames();
  auto color_frame = frames.get_color_frame();

  // Convert the frame to an OpenCV image
  cv::Mat frame(cv::Size(cam_res_width, cam_res_height), CV_8UC3, const_cast<void*>(color_frame.get_data()), cv::Mat::AUTO_STEP);

  // Detect the marker
  std::vector<int> markerIds;
  std::vector<std::vector<cv::Point2f>> markerCorners, rejectedCandidates;
  detector.detectMarkers(frame, markerCorners, markerIds, rejectedCandidates);

  size_t nMarkers = markerCorners.size();
  std::vector<Vec3d> rvecs(nMarkers), tvecs(nMarkers);

  if(estimatePose && !markerIds.empty()) {
    // Calculate pose for each marker
    for (size_t i = 0; i < nMarkers; i++) {
      bool success = solvePnP(objPoints, markerCorners.at(i), camMatrix, distCoeffs, rvecs.at(i), tvecs.at(i));
      if (success) {
        std::cout << "Rotation Vector: " << rvecs.at(i) << std::endl;
        std::cout << "Translation Vector: " << tvecs.at(i) << std::endl;
      } else {
        std::cerr << "Could not solve PnP problem." << std::endl;
      }
    }
  }

  // If the marker is detected, draw the marker
  if (!markerIds.empty()) {
    cv::aruco::drawDetectedMarkers(frame, markerCorners, markerIds);
    for(unsigned int i = 0; i < markerIds.size(); i++)
      cv::drawFrameAxes(frame, camMatrix, distCoeffs, rvecs[i], tvecs[i], markerLength * 1.5f, 2);
  }

  // Display the frame
  cv::imshow("Frame", frame);

  // If the user presses a key
  if (cv::waitKey(1) >= 0) {
    break;
  }
}
pipe.stop();
destroyAllWindows();
}

void findObjectPose(const std::vector<cv::Point3d>& objectPoints,
                    const std::vector<cv::Point2d>& imagePoints,
                    const cv::Mat& cameraMatrix,
                    const cv::Mat& distCoeffs,
                    cv::Vec3d& rvec,
                    cv::Vec3d& tvec) {
  // Use solvePnP to find the rotation and translation vectors
  bool success = cv::solvePnP(objectPoints, imagePoints, cameraMatrix, distCoeffs, rvec, tvec, false, cv::SOLVEPNP_ITERATIVE);

  if (success) {
    std::cout << "Rotation Vector: " << rvec << std::endl;
    std::cout << "Translation Vector: " << tvec << std::endl;
  } else {
    std::cerr << "Could not solve PnP problem." << std::endl;
  }
}




int main() {
  //generateMarker();
  //findMarkerAndPose();

  return 0;
}