#include <pose_estimation.h>

int main() {
    // Initial variables
    auto home_position = HomePosition::FRONT;
    std::string input_name = "spray_bottle.pcd";
    std::string output_directory = "../pose_estimation_cluster";
    constexpr bool debugging_cluster_filtering = false;
    constexpr bool debugging_movement_target = true;

    // Start the Realsense pipeline
    rs2::pipeline pipeline;
    rs2::config cfg;
    cfg.enable_stream(RS2_STREAM_DEPTH, cam_res_width, cam_res_height, RS2_FORMAT_Z16, cam_fps);
    //cfg.enable_stream(RS2_STREAM_ACCEL, RS2_FORMAT_MOTION_XYZ32F);
    pipeline.start(cfg);

    // Camera warmup - dropping several first frames to let auto-exposure stabilize
    for (int i = 0; i < 60; i++) {
        auto frames = pipeline.wait_for_frames();
    }
    // Only need to get accel data once
    /*rs2::frameset frames = pipeline.wait_for_frames();
    rs2::motion_frame accel_frame = frames.first_or_default(RS2_STREAM_ACCEL);
    rs2_vector accel_data = accel_frame.get_motion_data();*/

    // Variables to interact with the processing thread
    std::shared_ptr<std::vector<pcl::PointIndices>> cluster_indices(new std::vector<pcl::PointIndices>);
    std::shared_ptr<pcl::PointCloud<pcl::PointXYZ>> closest_point_pcl(new pcl::PointCloud<pcl::PointXYZ>);
    Eigen::Matrix4f closest_point = Eigen::Matrix4f::Zero();
    std::vector<vfh_model> models;
    std::vector<std::string> model_files; // Vector to store file names of the models
    std::string best_model_file; // Variable to store the best model file name
    std::mutex mtx;
    std::condition_variable condition_var;
    bool ready = false;

    // This thread will constantly capture data from the Realsense camera and produce a point cloud cluster
    // This point cloud cluster will be the cluster from the scene that is the best match to the stored models

    std::thread processing_thread(process_point_cloud_and_find_best_match, std::ref(pipeline),
        std::ref(cluster_indices), std::ref(closest_point_pcl), std::ref(best_model_file),
        debugging_cluster_filtering, debugging_movement_target,
        std::ref(mtx), std::ref(condition_var), std::ref(ready));

    PoseManager pose_manager;

    // Copies of the output variables from the processing thread
    std::string local_best_model_file;

    while (true) {
        {
            // Wait for the processing thread to produce a new point cloud cluster
            std::unique_lock lock(mtx);
            condition_var.wait(lock, [&ready] { return ready; });
            ready = false;

            // Extract the closest point Matrix from the pcl
            for (const auto& point : *closest_point_pcl) {
                closest_point(0) = point.x;
                closest_point(1) = point.y;
                closest_point(2) = point.z;
            }

            // Copy the values into local variables
            local_best_model_file = best_model_file;
        }
        Eigen::Matrix4f adjusted_point = closest_point;
        // Set the adjusted pose to a human-readable format (mm)
        adjusted_point(0) *= 1000;
        adjusted_point(1) *= 1000;
        adjusted_point(2) *= 1000;
        // Apply the pose manager's calibration adjustments to the adjusted point
        adjusted_point = adjusted_point * pose_manager.getPoseCalibrationAdjuster();
        if (local_best_model_file.find("spray_bottle") != std::string::npos)
        {
            adjusted_point(2) += spray_collision_offset; // Offset Z for the collision distance between the bottle and the manipulator
        }

        // Prompt the user to move the robot to the closest point
        std::cout << "In order to move the robot (end effector TCP) to the closest point, apply the following" << std::endl;
        std::cout << "All with respect to the TCP coordinates (select feature = Tool)" << std::endl;
        std::cout << "x: ";
        std::cout << std::fixed << std::setprecision(2) << adjusted_point(0) << std::endl;
        std::cout << "y: ";
        std::cout << std::fixed << std::setprecision(2) << -adjusted_point(1) << std::endl; // Flipped y-axis
        std::cout << "z: ";
        std::cout << std::fixed << std::setprecision(2) << adjusted_point(2) << std::endl;
        if (home_position == HomePosition::ELEVATED) {
            // If the home position is elevated, then the rotation needs to be applied
            std::cout << "RX (Rotation Vector degrees): ";
            std::cout << camera_rotation_home_position << std::endl;
        }
        std::cout << "Once done, press Enter to continue" << std::endl;
        // Wait for the user to press a key
        std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');

        // At this stage there would be touch sensors to give feedback if the robot has touched the object
        // In this circumstance, the update is manual
        std::cout << "Measure the error between the estimated pose and the actual pose." << std::endl;
        std::cout << "Record it as 'how much more the robot should move in the x,y,z direction' to be correct" << std::endl;
        std::cout << "Once done, press Enter to continue" << std::endl;
        // Wait for the user to press a key
        std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
        // Prompt user to return the robot to the original position
        std::cout << "Please reset the robot to the home position, then press Enter" << std::endl;
        std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
        // Prompt user to input the error in the x, y, z direction
        float x_error, y_error, z_error;
        std::cout << "Enter the error in the x direction (mm) eg. ' 2.1 ': ";
        std::cin >> x_error;
        std::cout << "Enter the error in the y direction (mm) eg. ' 1.2 ': ";
        std::cin >> y_error;
        std::cout << "Enter the error in the z direction (mm) eg. ' 1.1 ': ";
        std::cin >> z_error;
        // Update the pose manager with the error
        pose_manager.updatePoseCalibrationAdjusterXYZ(x_error / 1000, -y_error / 1000, z_error / 1000);
        // Output the pose calibration adjuster value
        std::cout << "The pose calibration adjuster is: " << std::endl;
        std::cout << pose_manager.getPoseCalibrationAdjuster() << std::endl;
    }
    processing_thread.join();
    pipeline.stop();
    return 0;
}

