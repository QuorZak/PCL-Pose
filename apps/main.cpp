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

    pcl::console::parse_argument(argc, argv, "-i", input_name);
    pcl::console::parse_argument(argc, argv, "-m", model_directory);
    pcl::console::parse_argument(argc, argv, "-k", k);
    pcl::console::parse_argument(argc, argv, "-t", thresh);

    // Load the VFH model library and build its search index
    // (before starting the camera and the thread, so there is nothing to shut down if the library is empty)
    const VfhLibrary library(model_directory);
    if (library.empty()) {
        std::cerr << "No VFH models found in " << model_directory << std::endl;
        return 1;
    }

    rs2::pipeline pipeline;
    rs2::config cfg;
    cfg.enable_stream(RS2_STREAM_COLOR, cam_res_width, cam_res_height, RS2_FORMAT_BGR8, cam_fps);
    cfg.enable_stream(RS2_STREAM_DEPTH, cam_res_width, cam_res_height, RS2_FORMAT_Z16, cam_fps);
    pipeline.start(cfg);

    auto profile = pipeline.get_active_profile();
    auto video_stream = profile.get_stream(RS2_STREAM_COLOR).as<rs2::video_stream_profile>();
    auto intrinsics = video_stream.get_intrinsics();

    // The pose is in the depth camera's frame, but it is drawn on the colour image, which comes from a different
    // imager. This moves the pose from the depth frame into the colour frame before it is projected.
    const rs2_extrinsics depth_to_color_extrinsics = profile.get_stream(RS2_STREAM_DEPTH).get_extrinsics_to(video_stream);
    Eigen::Matrix4f depth_to_color = Eigen::Matrix4f::Identity();
    for (int row = 0; row < 3; ++row) {
        for (int col = 0; col < 3; ++col) {
            depth_to_color(row, col) = depth_to_color_extrinsics.rotation[col * 3 + row]; // stored column-major
        }
        depth_to_color(row, 3) = depth_to_color_extrinsics.translation[row];
    }

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
        std::string best_model_file; // Stays empty until a cluster matches a model

        for (const auto& cluster : cluster_indices_local) {
            pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_cluster(new pcl::PointCloud<pcl::PointXYZ>);
            for (const auto& idx : cluster.indices) {
                cloud_cluster->push_back(output_stream_cloud_local->points[idx]);
            }

            pcl::PointCloud<pcl::VFHSignature308> signature;
            estimate_VFH(cloud_cluster, signature);

            vfh_model query_model;
            query_model.second.assign(signature.points[0].histogram, signature.points[0].histogram + 308);
            const std::vector<VfhLibrary::Match> matches = library.nearest(query_model, k);

            if (!matches.empty() && matches[0].distance < best_distance) {
                best_distance = matches[0].distance;
                best_model_file = matches[0].file;
                *best_cluster = *cloud_cluster;
            }
        }

        if (!best_model_file.empty()) {
            // Load the library cloud that belongs to the best matching VFH (the cloud file has the same name without _vfh)
            const std::string model_cloud_file = cloudFileForVfh(best_model_file);
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
            displayCoordinateSystem(depth_to_color * stored_pose, frame, intrinsics);
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
