#include <pose_estimation.h>

int main(int argc, char** argv) {
    std::string input_name = "capture.pcd";
    std::string output_directory = "../pose_estimation_cluster";
    int k = 6;
    auto thresh = DBL_MAX;

    std::vector<vfh_model> models;
    Matrix<int> k_indices;
    Matrix<float> k_distances;

    pcl::console::parse_argument(argc, argv, "-i", input_name);
    pcl::console::parse_argument(argc, argv, "-m", model_directory);
    pcl::console::parse_argument(argc, argv, "-k", k);
    pcl::console::parse_argument(argc, argv, "-t", thresh);

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

    pcl::PointCloud<pcl::PointXYZ>::Ptr output_stream_cloud(new pcl::PointCloud<pcl::PointXYZ>);
    std::shared_ptr<std::vector<pcl::PointIndices>> cluster_indices(new std::vector<pcl::PointIndices>);
    std::mutex mtx;
    std::condition_variable condition_var;
    bool ready = false;

    std::thread processing_thread(processPointCloud, std::ref(pipeline), std::ref(output_stream_cloud), std::ref(cluster_indices), std::ref(mtx), std::ref(condition_var), std::ref(ready));

    load_vfh_model_data(model_directory, models);

    std::unique_ptr<float[]> data_ptr(new float[models.size() * models[0].second.size()]);
    Matrix data(data_ptr.get(), models.size(), models[0].second.size());
    for (size_t i = 0; i < data.rows; ++i)
        for (size_t j = 0; j < data.cols; ++j)
            data[i][j] = models[i].second[j];

    Index<ChiSquareDistance<float>> index(data, LinearIndexParams());
    index.buildIndex();

    PoseManager pose_manager;

    while (true) {
        {
            std::unique_lock<std::mutex> lock(mtx);
            condition_var.wait(lock, [&ready] { return ready; });
            ready = false;
        }

        pcl::PointCloud<pcl::PointXYZ>::Ptr best_cluster(new pcl::PointCloud<pcl::PointXYZ>);
        float best_distance = std::numeric_limits<float>::max();
        int best_index = -1;

        for (const auto& cluster : *cluster_indices) {
            pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_cluster(new pcl::PointCloud<pcl::PointXYZ>);
            for (const auto& idx : cluster.indices) {
                cloud_cluster->push_back(output_stream_cloud->points[idx]);
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
            Eigen::Matrix4f new_pose = estimatePose(output_stream_cloud, best_cluster);
            pose_manager.updatePose(new_pose, true);
            Eigen::Matrix4f stored_pose = pose_manager.getPose();
            std::cout << "Updated Pose:\n" << stored_pose << std::endl;

            rs2::frameset frames = pipeline.wait_for_frames();
            rs2::frame color_frame = frames.get_color_frame();
            cv::Mat frame(cv::Size(640, 480), CV_8UC3, const_cast<void*>(color_frame.get_data()), cv::Mat::AUTO_STEP);
            displayCoordinateSystem(stored_pose, frame, intrinsics);
            cv::imshow("Coordinate System", frame);
            if (cv::waitKey(1) >= 0) {
                pipeline.stop();
                break;
            }
        }
    }
    processing_thread.join();
    return 0;
}