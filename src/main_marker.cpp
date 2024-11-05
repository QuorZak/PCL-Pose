#include <marker_pose_estimation.h>

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
  Mat camMatrix = (cv::Mat_<double>(3, 3) << intrinsics.fx, 0, intrinsics.ppx, 0, intrinsics.fy, intrinsics.ppy, 0, 0, 1);
  Mat distCoeffs = (cv::Mat_<double>(5, 1) << intrinsics.coeffs[0], intrinsics.coeffs[1], intrinsics.coeffs[2], intrinsics.coeffs[3], intrinsics.coeffs[4]);

  // set coordinate system
  float markerLength = 0.1f;
  Mat objPoints(4, 1, CV_32FC3);
  objPoints.ptr<Vec3f>(0)[0] = Vec3f(-markerLength/2.f, markerLength/2.f, 0);
  objPoints.ptr<Vec3f>(0)[1] = Vec3f(markerLength/2.f, markerLength/2.f, 0);
  objPoints.ptr<Vec3f>(0)[2] = Vec3f(markerLength/2.f, -markerLength/2.f, 0);
  objPoints.ptr<Vec3f>(0)[3] = Vec3f(-markerLength/2.f, -markerLength/2.f, 0);

  // Create the detector
  auto detectorParams = aruco::DetectorParameters();
  aruco::Dictionary dictionary = aruco::getPredefinedDictionary(aruco_dict_number);
  aruco::ArucoDetector detector(dictionary, detectorParams);

  while (true) {
    // Get the frames
    auto frames = pipe.wait_for_frames();
    auto color_frame = frames.get_color_frame();

    // Convert the frame to an OpenCV image
    Mat frame(Size(cam_res_width, cam_res_height), CV_8UC3, const_cast<void*>(color_frame.get_data()), Mat::AUTO_STEP);

    // Detect the marker
    std::vector<int> markerIds;
    std::vector<std::vector<Point2f>> markerCorners, rejectedCandidates;
    detector.detectMarkers(frame, markerCorners, markerIds, rejectedCandidates);

    size_t nMarkers = markerCorners.size();
    std::vector<Vec3d> rot_vects(nMarkers), trans_vects(nMarkers);

    // create 6d pose variables
    std::vector<Eigen::Matrix4f> poses(nMarkers);

    if(!markerIds.empty()) {
      // Calculate pose for each marker
      for (size_t i = 0; i < nMarkers; i++) {
        solvePnP(objPoints, markerCorners.at(i), camMatrix, distCoeffs, rot_vects.at(i), trans_vects.at(i));

        // Take rot and trans and make a 6d pose
        Eigen::MatrixXd rot_mat;
        cv::Mat rot_mat_cv;
        cv::Rodrigues(rot_vects.at(i), rot_mat_cv);
        cv::cv2eigen(rot_mat_cv, rot_mat);
        poses[i].setIdentity();
        poses[i].block<3, 3>(0, 0) = rot_mat.cast<float>();
        poses[i].block<3, 1>(0, 3) = Eigen::Vector3f(trans_vects[i][0], trans_vects[i][1], trans_vects[i][2]);

        std::cout << "Pose: " << poses[i] << std::endl;
      }
    }

    // If the marker is detected, draw the marker
    if (!markerIds.empty()) {
      aruco::drawDetectedMarkers(frame, markerCorners, markerIds);
      for(unsigned int i = 0; i < markerIds.size(); i++)
        drawFrameAxes(frame, camMatrix, distCoeffs, rot_vects[i], trans_vects[i], markerLength * 1.5f, 2);
    }

    // Display the frame
    imshow("Frame", frame);

    // If the user presses a key
    if (waitKey(1) >= 0) {
      break;
    }
  }
  pipe.stop();
  destroyAllWindows();
}


int main() {
  // Start camera and find the markers
  findMarkerAndPose();

  return 0;
}