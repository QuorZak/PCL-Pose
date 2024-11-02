#include <pose_estimation.h>

int main() {
    // Initial variables
    std::string input_name = "spray_bottle.pcd";
    std::string output_directory = "../pose_estimation_cluster";
    int k_val = 6;

    std::vector<vfh_model> models;
    std::vector<std::string> model_files; // Vector to store file names
    Matrix<int> k_indices;
    Matrix<float> k_distances;

    // Start the Realsense pipeline
    rs2::pipeline pipeline;
    rs2::config cfg;
    cfg.enable_stream(RS2_STREAM_DEPTH, cam_res_width, cam_res_height, RS2_FORMAT_Z16, cam_fps);
    pipeline.start(cfg);

    // Camera warmup - dropping several first frames to let auto-exposure stabilize
    for (int i = 0; i < 60; i++) {
        auto frames = pipeline.wait_for_frames();
    }

    pcl::PointCloud<pcl::PointXYZ>::Ptr output_stream_cloud(new pcl::PointCloud<pcl::PointXYZ>);
    std::shared_ptr<std::vector<pcl::PointIndices>> cluster_indices(new std::vector<pcl::PointIndices>);
    std::mutex mtx;
    std::condition_variable condition_var;
    bool ready = false;
    constexpr bool debugging = false;

    // This thread will constantly capture data from the Realsense camera and produce clustered point clouds
    std::thread processing_thread(processPointCloud, std::ref(pipeline), std::ref(output_stream_cloud),
        std::ref(cluster_indices), std::ref(debugging), std::ref(mtx), std::ref(condition_var), std::ref(ready));

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

    PoseManager pose_manager;

    while (true) {
        {
            // Wait for the processing thread to produce a new point cloud cluster
            std::unique_lock<std::mutex> lock(mtx);
            condition_var.wait(lock, [&ready] { return ready; });
            ready = false;
        }
        // Initialise the best cluster variables
        pcl::PointCloud<pcl::PointXYZ>::Ptr best_cluster(new pcl::PointCloud<pcl::PointXYZ>);
        pcl::PointCloud<pcl::PointXYZ>::Ptr best_cluster_pose_from_file(new pcl::PointCloud<pcl::PointXYZ>);
        float best_distance = std::numeric_limits<float>::max();
        int best_index = -1;

        // Iterate through all the clusters and find the best match of a scene cluster to a stored model via VFH signatures
        for (const auto& cluster : *cluster_indices) {
            pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_cluster(new pcl::PointCloud<pcl::PointXYZ>);
            for (const auto& idx : cluster.indices) {
                cloud_cluster->push_back(output_stream_cloud->points[idx]);
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
        // If a best match is found, continue to process the data
        if (best_index != -1) {
            // Load the corresponding file from the model_directory
            const std::string& best_model_file = model_files[best_index];
            std::cout << "Best match found: " << best_model_file << std::endl;
            pcl::io::loadPCDFile(best_model_file, *best_cluster_pose_from_file);

            // Rotate the stored pose to

            // Transform the reference the camera frame to the robot frame (use camera_to_robot_frame)
            pcl::PointCloud<pcl::PointXYZ>::Ptr object_wrt_robot(new pcl::PointCloud<pcl::PointXYZ>);
            pcl::transformPointCloud(*best_cluster, *object_wrt_robot, camera_to_robot_frame);

            // Trim the point cloud to get a band around the lower part of the object
            pcl::PointCloud<pcl::PointXYZ>::Ptr object_band(new pcl::PointCloud<pcl::PointXYZ>);
            for (const auto& point : object_wrt_robot->points) {
                if (point.y < 0.50 * object_wrt_robot->height && point.y > 0.00 * object_wrt_robot->height) {
                    object_band->push_back(point);
                }
            }
            // visualise the point cloud of the object band
            pcl::visualization::PCLVisualizer viewer("Object Band");
            viewer.addPointCloud<pcl::PointXYZ>(object_band, "object_band");
            // add coordinate system
            viewer.addCoordinateSystem(0.1);
            viewer.spin();
            // wait for the user to press a key
            std::cout << "Press Enter to continue" << std::endl;
            std::cin.get();

            // Find the closest point to the robot
            Eigen::Matrix4f closest_point;
            pcl::PointXYZ min_p, max_p;
            pcl::getMinMax3D(*object_band, min_p, max_p);
            // Store the closest point
            closest_point(0) = min_p.x;
            closest_point(1) = min_p.y;
            closest_point(2) = min_p.z;
            closest_point(3) = 1;

            // If it is not the first pose estimate,
            // Apply the pose manager's pose adjustments to the closest point
            Eigen::Matrix4f adjusted_point;
            if (!pose_manager.getPose().isIdentity()) {
                pose_manager.updatePose(closest_point);
                adjusted_point = pose_manager.getPose();
            } else {
                // If it is the first pose estimate, set the pose manager's pose to the closest point
                pose_manager.setPose(closest_point);
                adjusted_point = closest_point;
            }
            // Set the adjusted pose to a human-readable format (mm)
            adjusted_point(0) *= 1000;
            adjusted_point(1) *= 1000;
            adjusted_point(2) *= 1000;
            if (input_name == "spray_bottle.pcd")
            {
                adjusted_point(2) += 60.0f; // Offset Z for the collision distance between the bottle and the manipulator
            }

            // Prompt the user to move the robot to the closest point
            std::cout << "In order to move the robot to the closest point, apply the following" << std::endl;
            std::cout << "Rotation-x (w.r.t tool): ";
            std::cout << camera_rotation_home_position << std::endl;
            std::cout << "Once done, press Enter to continue" << std::endl;
            // Wait for the user to press a key
            std::cin.get();
            std::cout << "Now set the following coordinates" << std::endl;
            std::cout << "x: ";
            std::cout << std::fixed << std::setprecision(2) << adjusted_point(0) << std::endl;
            std::cout << "y: ";
            std::cout << std::fixed << std::setprecision(2) << -adjusted_point(1) << std::endl; // Flipped y-axis
            std::cout << "z: ";
            std::cout << std::fixed << std::setprecision(2) << adjusted_point(2) << std::endl;
            // And rotation from the home position
            std::cout << "Once done, press Enter to continue" << std::endl;
            // Wait for the user to press a key
            std::cin.get();

        }
    }
    processing_thread.join();
    pipeline.stop();
    return 0;
}