#include <pose_estimation.h>

using namespace pcl_pose;

int main(int argc, char** argv) {
    if (!loadConfigFromArguments(argc, argv)) {
        return 1;
    }
    std::string input_name = "capture.pcd";
    constexpr bool debugging_cluster_filtering = false;
    int k = 6;
    auto thresh = DBL_MAX;

    std::vector<vfh_model> models;
    flann::Matrix<int> k_indices;
    flann::Matrix<float> k_distances;

    pcl::console::parse_argument(argc, argv, "-i", input_name);
    pcl::console::parse_argument(argc, argv, "-m", model_directory);
    pcl::console::parse_argument(argc, argv, "-k", k);
    pcl::console::parse_argument(argc, argv, "-t", thresh);

    // Load the VFH model data and store the names of the models in the order they are in the index
    // (before starting the camera and the thread, so there is nothing to shut down if the library is empty)
    std::vector<std::string> model_files;
    load_vfh_model_data(model_directory, models, model_files);
    if (models.empty()) {
        std::cerr << "No VFH models found in " << model_directory << std::endl;
        return 1;
    }

    std::unique_ptr<float[]> data_ptr(new float[models.size() * models[0].second.size()]);
    flann::Matrix data(data_ptr.get(), models.size(), models[0].second.size());
    for (size_t i = 0; i < data.rows; ++i)
        for (size_t j = 0; j < data.cols; ++j)
            data[i][j] = models[i].second[j];

    flann::Index<flann::ChiSquareDistance<float>> index(data, flann::LinearIndexParams());
    index.buildIndex();

    rs2::pipeline pipeline;
    rs2::config cfg;
    cfg.enable_stream(RS2_STREAM_COLOR, cam_res_width, cam_res_height, RS2_FORMAT_BGR8, cam_fps);
    cfg.enable_stream(RS2_STREAM_DEPTH, cam_res_width, cam_res_height, RS2_FORMAT_Z16, cam_fps);
    pipeline.start(cfg);

    auto profile = pipeline.get_active_profile();
    auto video_stream = profile.get_stream(RS2_STREAM_COLOR).as<rs2::video_stream_profile>();
    auto intrinsics = video_stream.get_intrinsics();

    for (int i = 0; i < 100; i++) {
        auto frames = pipeline.wait_for_frames();
    }

    std::shared_ptr<pcl::PointCloud<pcl::PointXYZ>> output_stream_cloud(new pcl::PointCloud<pcl::PointXYZ>);
    std::shared_ptr<std::vector<pcl::PointIndices>> cluster_indices(new std::vector<pcl::PointIndices>);
    std::mutex mtx;
    std::condition_variable condition_var;
    bool ready = false;
    std::atomic<bool> stop_requested{false};

    // This thread will constantly capture data from the Realsense camera and produce clustered point clouds
    // (no accelerometer data is needed here, hence nullptr)
    std::thread processing_thread(processPointCloud, std::ref(pipeline), std::ref(output_stream_cloud),
        std::ref(cluster_indices), nullptr, std::ref(debugging_cluster_filtering), std::ref(mtx),
        std::ref(condition_var), std::ref(ready), std::ref(stop_requested));

    PoseManager pose_manager;

    // Get copy from the shared pointers
    pcl::PointCloud<pcl::PointXYZ>::Ptr output_stream_cloud_local(new pcl::PointCloud<pcl::PointXYZ>);
    std::vector<pcl::PointIndices> cluster_indices_local;

    while (true) {
        {
            // Wait for the processing thread to produce a new point cloud, and copy it while it is not being written to
            std::unique_lock<std::mutex> lock(mtx);
            condition_var.wait(lock, [&ready] { return ready; });
            *output_stream_cloud_local = *output_stream_cloud;
            cluster_indices_local = *cluster_indices;
            ready = false;
        }

        pcl::PointCloud<pcl::PointXYZ>::Ptr best_cluster(new pcl::PointCloud<pcl::PointXYZ>);
        float best_distance = std::numeric_limits<float>::max();
        int best_index = -1;

        for (const auto& cluster : cluster_indices_local) {
            pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_cluster(new pcl::PointCloud<pcl::PointXYZ>);
            for (const auto& idx : cluster.indices) {
                cloud_cluster->push_back(output_stream_cloud_local->points[idx]);
            }

            pcl::PointCloud<pcl::VFHSignature308> signature;
            estimate_VFH(cloud_cluster, signature);

            vfh_model query_model;
            query_model.second.assign(signature.points[0].histogram, signature.points[0].histogram + 308);
            nearestKSearch(index, query_model, k, k_indices, k_distances);

            if (k_distances[0][0] < best_distance) {
                best_distance = k_distances[0][0];
                best_index = k_indices[0][0];
                *best_cluster = *cloud_cluster;
            }
        }

        if (best_index != -1) {
            // Load the library cloud that belongs to the best matching VFH (the cloud file has the same name without _vfh)
            std::string model_cloud_file = model_files[best_index];
            model_cloud_file = model_cloud_file.substr(0, model_cloud_file.find("_vfh.pcd")) + ".pcd";
            pcl::PointCloud<pcl::PointXYZ>::Ptr model_cloud(new pcl::PointCloud<pcl::PointXYZ>);
            if (pcl::io::loadPCDFile(model_cloud_file, *model_cloud) == -1) {
                std::cerr << "Couldn't read file " << model_cloud_file << std::endl;
                continue;
            }

            // Align the library model to the matched cluster in the scene, which gives the pose of the object
            Eigen::Matrix4f new_pose = estimatePose(best_cluster, model_cloud);
            pose_manager.updatePose(new_pose, true);
            Eigen::Matrix4f stored_pose = pose_manager.getPose();
            std::cout << "Updated Pose:\n" << stored_pose << std::endl;

            // Known limitation: the processing thread is also reading frames from this pipeline, so each of the
            // two only gets some of the framesets (the colour image may be a little out of date or skipped)
            rs2::frameset frames = pipeline.wait_for_frames();
            rs2::frame color_frame = frames.get_color_frame();
            cv::Mat frame(cv::Size(cam_res_width, cam_res_height), CV_8UC3, const_cast<void*>(color_frame.get_data()),
                cv::Mat::AUTO_STEP);
            displayCoordinateSystem(stored_pose, frame, intrinsics);
            cv::imshow("Coordinate System", frame);
            if (cv::waitKey(1) >= 0) {
                break;
            }
        }
    }
    // Ask the processing thread to finish and wait for it, and only then stop the camera
    stop_requested = true;
    processing_thread.join();
    pipeline.stop();
    cv::destroyAllWindows();
    return 0;
}
