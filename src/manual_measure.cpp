#include <pose_estimation.h>

int main() {
    // Initial variables
    auto home_position = HomePosition::ANY;
    std::string input_name = "spray_bottle.pcd";
    std::string output_directory = "../pose_estimation_cluster";
    constexpr bool debugging_cluster_filtering = false;
    constexpr bool debugging_target_acquire = false;
    constexpr bool use_averaging = false; // not working correctly at the moment
    constexpr bool do_calibration = false;
    constexpr int k_val = 6;
    constexpr int closest_point_averaging_buffer_size = 10;
    Eigen::MatrixXf closest_point_buffer(3, closest_point_averaging_buffer_size);

    std::vector<vfh_model> models;
    std::vector<std::string> model_files; // Vector to store file names
    Matrix<int> k_indices;
    Matrix<float> k_distances;

    // Start the Realsense pipeline
    rs2::pipeline pipeline;
    rs2::config cfg;
    rs2_vector accel_data;
    float rx_axis_rotation_degrees = 0.0f;
    cfg.enable_stream(RS2_STREAM_DEPTH, cam_res_width, cam_res_height, RS2_FORMAT_Z16, cam_fps);
    if (home_position == HomePosition::ELEVATED || home_position == HomePosition::ANY) {
        cfg.enable_stream(RS2_STREAM_ACCEL, RS2_FORMAT_MOTION_XYZ32F);
    }

    pipeline.start(cfg);

    // Camera warmup - dropping several first frames to let auto-exposure stabilize
    for (int i = 0; i < 60; i++) {
        auto frames = pipeline.wait_for_frames();
    }

    std::shared_ptr<pcl::PointCloud<pcl::PointXYZ>> output_stream_cloud(new pcl::PointCloud<pcl::PointXYZ>);
    std::shared_ptr<std::vector<pcl::PointIndices>> cluster_indices(new std::vector<pcl::PointIndices>);
    std::mutex mtx;
    std::condition_variable condition_var;
    bool ready = false;

    // This thread will constantly capture data from the Realsense camera and produce clustered point clouds
    std::thread processing_thread(processPointCloud, std::ref(pipeline), std::ref(output_stream_cloud),
        std::ref(cluster_indices), std::ref(debugging_cluster_filtering), std::ref(mtx), std::ref(condition_var), std::ref(ready));

    // Load the VFH model data and store the names of the models in the order they are in the index
    load_vfh_model_data(model_directory, models, model_files);
    // Convert the models to FLANN format
    std::unique_ptr<float[]> data_ptr(new float[models.size() * models[0].second.size()]);
    Matrix data(data_ptr.get(), models.size(), models[0].second.size());
    for (size_t i = 0; i < data.rows; ++i)
        for (size_t j = 0; j < data.cols; ++j)
            data[i][j] = models[i].second[j];
    // Build the FLANN index. This is the data structure that will be used to search for the best match
    Index<ChiSquareDistance<float>> index(data, LinearIndexParams());
    index.buildIndex();
    std::cout << "VFH model data loaded and index built" << std::endl;

    PoseManager pose_manager;

    // Get copy from the shared pointers
    pcl::PointCloud<pcl::PointXYZ>::Ptr output_stream_cloud_local(new pcl::PointCloud<pcl::PointXYZ>);
    std::vector<pcl::PointIndices> cluster_indices_local;

    while (true) {
        {
            // Wait for the processing thread to produce a new point cloud cluster
            std::unique_lock<std::mutex> lock(mtx);
            condition_var.wait(lock, [&ready] { return ready; });
            *output_stream_cloud_local = *output_stream_cloud;
            cluster_indices_local = *cluster_indices;
            ready = false;
        }
        if (home_position == HomePosition::ELEVATED || home_position == HomePosition::ANY)
        {
            // Get the current camera pose accel data
            rs2::frameset frames = pipeline.wait_for_frames();
            rs2::motion_frame accel_frame = frames.first_or_default(RS2_STREAM_ACCEL);
            accel_data = accel_frame.get_motion_data();
        }

        // Initialise the best cluster variables
        pcl::PointCloud<pcl::PointXYZ>::Ptr best_cluster(new pcl::PointCloud<pcl::PointXYZ>);
        pcl::PointCloud<pcl::PointXYZ>::Ptr best_cluster_pose_from_file(new pcl::PointCloud<pcl::PointXYZ>);
        float best_distance = std::numeric_limits<float>::max();
        int best_index = -1;

        // Iterate through all the clusters and find the best match of a scene cluster to a stored model via VFH signatures
        for (const auto& cluster : cluster_indices_local) {
            pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_cluster(new pcl::PointCloud<pcl::PointXYZ>);
            for (const auto& idx : cluster.indices) {
                cloud_cluster->push_back(output_stream_cloud_local->points[idx]);
            }
            pcl::PointCloud<pcl::VFHSignature308> signature;
            estimate_VFH(cloud_cluster, signature);
            vfh_model query_model;
            query_model.second.assign(signature.points[0].histogram, signature.points[0].histogram + 308);
            nearestKSearch(index, query_model, k_val, k_indices, k_distances);
            // Store the best match
            if (k_distances[0][0] < best_distance) {
                best_distance = k_distances[0][0];
                best_index = k_indices[0][0];
                *best_cluster = *cloud_cluster;
            }
        }
        // If the best match is found, continue to process the data
        if (best_index != -1) {
            // Load the corresponding file from the model_directory
            const std::string& best_model_file = model_files[best_index];
            std::cout << "Best pose match found: " << best_model_file << std::endl;
            //pcl::io::loadPCDFile(best_model_file, *best_cluster_pose_from_file);

            // Transform the reference the camera frame to the robot end effector frame (TCP)
            pcl::PointCloud<pcl::PointXYZ>::Ptr object_wrt_TCP(new pcl::PointCloud<pcl::PointXYZ>);
            transformPointCloud(*best_cluster, *object_wrt_TCP, camera_to_TCP);

            // Manage the rotation of the camera frame to the robot end effector frame (TCP) (if any)
            if (home_position == HomePosition::ELEVATED || home_position == HomePosition::ANY)
            {
                // Rotate the point cloud to align with gravity
                pcl::PointCloud<pcl::PointXYZ>::Ptr object_wrt_TCP_aligned(new pcl::PointCloud<pcl::PointXYZ>);
                std::unique_ptr<Eigen::Matrix4f> y_align_rotation(new Eigen::Matrix4f());
                align_yaxis_with_gravity(object_wrt_TCP, *y_align_rotation, accel_data, rx_axis_rotation_degrees);
                // Set the object_wrt_TCP to the aligned point cloud
                // object_wrt_TCP = object_wrt_TCP_aligned;
            }

            // Trim the point cloud to get a center target around the middle lower part of the object
            pcl::PointCloud<pcl::PointXYZ>::Ptr object_band(new pcl::PointCloud<pcl::PointXYZ>);
            // Get the middle (y) of the object
            float y_min = std::numeric_limits<float>::max();
            float y_max = std::numeric_limits<float>::min();
            for (const auto& point : object_wrt_TCP->points) {
                if (point.y < y_min) {
                    y_min = point.y;
                }
                if (point.y > y_max) {
                    y_max = point.y;
                }
            }
            // Get the middle (x) of the object
            float x_sum = 0;
            float x_min = std::numeric_limits<float>::max();
            float x_max = std::numeric_limits<float>::min();
            for (const auto& point : object_wrt_TCP->points) {
                x_sum += point.x;
                if (point.x < x_min) {
                    x_min = point.x;
                }
                if (point.x > x_max) {
                    x_max = point.x;
                }
            }
            float x_mean = x_sum / object_wrt_TCP->size();
            const float object_height = (y_max - y_min);
            const float object_width = (x_max - x_min);
            // Y calculated this way for always targeting below halfway
            const float object_height_center = y_min + (object_height / 2);
            const float object_band_start = object_height_center + (object_height/2.0f * 0.30f);
            const float object_band_stop = object_height_center + (object_height/2.0f * 0.70f);
            const float object_band_width = object_width * 0.25f; // Can adjust the percentage here

            // Get the points that are average or above the middle of the object (because y is flipped we want the bottom part)
            // Also as we are iterating through the points, store the closest one (in the z direction) to the camera
            // Trim any points that are on the sides of the object
            Eigen::Matrix4f closest_point;
            float closest_distance = std::numeric_limits<float>::max();
            for (const auto& point : object_wrt_TCP->points) {
                // only consider the center target area
                if (point.y >= object_band_start && point.y <= object_band_stop) {
                    if (point.x >= x_mean - object_band_width && point.x <= x_mean + object_band_width)
                    {
                        object_band->push_back(point);
                        if (point.z < closest_distance) {
                            closest_distance = point.z;
                            closest_point(0) = point.x;
                            closest_point(1) = point.y;
                            closest_point(2) = point.z;
                        }
                    }
                }
            }
            if (debugging_target_acquire)
            {
                object_band->width = object_band->size();
                object_band->height = 1;
                object_band->is_dense = false;

                // output the count of points in the object band
                std::cout << "Object band has " << object_band->size() << " points" << std::endl;
                // visualise the point cloud of the object band
                pcl::visualization::PCLVisualizer viewer("Object Band");
                viewer.addPointCloud<pcl::PointXYZ>(object_band, "object_band");
                // add coordinate system, then set the camera position towards the centroid of the object
                viewer.addCoordinateSystem(0.1);
                viewer.spin();
                while(true) {
                    // wait for the user to press a key
                    std::cout << "Press Enter to continue" << std::endl;
                    // if user presses Enter, break out of the loop
                    if (std::cin.get() == '\n') {
                        break;
                    }
                }
            }
            if (object_band->empty() || closest_distance == std::numeric_limits<float>::max()) {
                std::cout << "No points in the object band" << std::endl;
                continue;
            }
            if (use_averaging) // Isn't working at time of comment
            {
                // Get a series of samples of the closest point to the camera before progressing
                // Add the point to the buffer, then check if the count is reached
                closest_point_buffer.block<3, closest_point_averaging_buffer_size - 1>(0, 0) = closest_point_buffer.block<3, closest_point_averaging_buffer_size - 1>(0, 1);
                Eigen::Vector3f buffer_input;
                buffer_input(0) = closest_point(0);
                buffer_input(1) = closest_point(1);
                buffer_input(2) = closest_point(2);
                closest_point_buffer.block<3, 1>(0, closest_point_averaging_buffer_size - 1) = buffer_input;
                // If the count is reached, then average the points, clear the buffer and continue
                if (closest_point_buffer.cols() == closest_point_averaging_buffer_size) {
                    Eigen::Vector3f average_closest_point = closest_point_buffer.rowwise().mean();
                    closest_point(0) = average_closest_point(0);
                    closest_point(1) = average_closest_point(1);
                    closest_point(2) = average_closest_point(2);
                    closest_point_buffer.setZero();
                    std::cout << "Closest point averaged" << std::endl;
                } else
                {
                    continue;
                }
            }

            Eigen::Matrix4f adjusted_point = closest_point;
            // Set the adjusted pose to a human-readable format (mm)
            adjusted_point(0) *= 1000;
            adjusted_point(1) *= 1000;
            adjusted_point(2) *= 1000;
            // Apply the pose manager's calibration adjustments to the adjusted point
            adjusted_point = adjusted_point * pose_manager.getPoseCalibrationAdjuster();
            std::cout << "Closest point detected at: " << std::endl;
            std::cout << "x: " << adjusted_point(0) << " mm" << std::endl;
            std::cout << "y: " << adjusted_point(1) << " mm" << std::endl;
            std::cout << "z: " << adjusted_point(2) << " mm" << std::endl;
            if (input_name == "spray_bottle.pcd")
            {
                adjusted_point(2) += spray_collision_offset; // Offset Z for the collision distance between the bottle and the manipulator
            }

            // Prompt the user to move the robot to the closest point
            std::cout << "In order to move the robot (end effector TCP) to the closest point, apply the following" << std::endl;
            std::cout << "All with respect to the TCP coordinates (select feature = Tool)" << std::endl;
            std::cout << "x: ";
            std::cout << std::fixed << std::setprecision(2) << adjusted_point(0) << std::endl;
            std::cout << "y: ";
            std::cout << std::fixed << std::setprecision(2) << adjusted_point(1) << std::endl;
            std::cout << "z: ";
            std::cout << std::fixed << std::setprecision(2) << adjusted_point(2) << std::endl;
            if (home_position == HomePosition::ELEVATED || home_position == HomePosition::ANY
                && rx_axis_rotation_degrees >= 1.0f) {
                // If the home position is elevated, then the rotation needs to be applied
                std::cout << "RX (Rotation Vector degrees): ";
                std::cout << std::fixed << std::setprecision(1) << rx_axis_rotation_degrees << std::endl;
                //std::cout << camera_rotation_home_position << std::endl;
            }
            std::cout << "Once done, press Enter to continue" << std::endl;
            // Wait for the user to press a key
            std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');

            if (do_calibration)
            {
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
                std::cout << "Enter the error in the x direction (mm) eg. ' 2.0 ': ";
                std::cin >> x_error;
                std::cout << "Enter the error in the y direction (mm) eg. ' 1.2 ': ";
                std::cin >> y_error;
                std::cout << "Enter the error in the z direction (mm) eg. ' 1.1 ': ";
                std::cin >> z_error;
                // Update the pose manager with the error
                pose_manager.updatePoseCalibrationAdjusterXYZ(x_error / 1000, y_error / 1000, z_error / 1000);
                // Output the new pose calibration adjuster
                std::cout << "The new pose calibration adjuster is: " << std::endl;
            }
        }
    }
    // Stop the processing tread and the Realsense pipeline
    processing_thread.join();
    pipeline.stop();
    return 0;
}