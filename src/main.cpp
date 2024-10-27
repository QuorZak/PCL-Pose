#include <pose_estimation.h>

int main(int argc, char** argv) {
    std::string input_name = "capture.pcd";
    std::string output_directory = "../pose_estimation_cluster";
    int k = 6;
    auto thresh = DBL_MAX;

    std::vector<vfh_model> models;
    flann::Matrix<int> k_indices;
    flann::Matrix<float> k_distances;

    pcl::PointCloud<pcl::Normal> normals;
    pcl::PointCloud<pcl::VFHSignature308> vfh_descriptors;

    // Parse console inputs
    pcl::console::parse_argument(argc, argv, "-i", input_name);
    pcl::console::parse_argument(argc, argv, "-m", model_directory);
    pcl::console::parse_argument(argc, argv, "-k", k);
    pcl::console::parse_argument(argc, argv, "-t", thresh);

    rs2::pipeline pipeline;
    rs2::frame depth;
    rs2::points points;

    rs2::pipeline pipe;
    pcl::PointCloud<pcl::PointXYZ>::Ptr output_stream_cloud(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointCloud<pcl::PointXYZ> output_stream_copy;
    pcl::PointCloud<pcl::PointXYZ>::Ptr pcl_cloud(new pcl::PointCloud<pcl::PointXYZ>);
    std::mutex mtx;
    std::condition_variable condition_var;
    bool ready = false;

    // Start the streamDepthMap function in a separate thread
    cout << "Starting the point cloud streaming thread." << endl;
    std::thread img_thread(stream_point_cloud_show_depth_map, std::ref(pipe), std::ref(output_stream_cloud),
        std::ref(mtx), std::ref(condition_var), std::ref(ready));

    // Wait for the first available output Realsense cloud
    {
        std::unique_lock<std::mutex> lock(mtx);
        condition_var.wait(lock, [&ready] { return ready; });
        // Once the lock is acquired, copy the output stream cloud to the pcl cloud
        output_stream_copy = *output_stream_cloud;
    }
    // Create a usable pointer to the point cloud
    *pcl_cloud = output_stream_copy;

    // Ensure the point cloud dimensions are set correctly
    pcl_cloud->width = static_cast<uint32_t>(pcl_cloud->points.size());
    pcl_cloud->height = 1; // Since this is an unorganised point cloud
    pcl_cloud->is_dense = false;

    // Save the point cloud to a PCD file
    if (pcl_cloud->points.empty()) {
        std::cerr << "No points in the point cloud." << std::endl;
        return -1;
    }
    std::string pcd_path = output_directory + "/capture.pcd";
    pcl::io::savePCDFileASCII(pcd_path, *pcl_cloud);
    std::cout << "Saved " << pcl_cloud->points.size() << " data points to " << pcd_path << std::endl;

    // Now we're finished getting the point cloud, so we can move on to getting the VFH
    std::cout << "Finished getting point cloud, now getting VFH." << std::endl;
    vfh_model histogram;
    std::string vfh_path = output_directory + "/capture_vfh.pcd";

    // Estimate signature and save it in a file
    pcl::PointCloud<pcl::VFHSignature308> signature;
    estimate_VFH(pcl_cloud, signature);
    pcl::io::savePCDFile(vfh_path, signature);
    std::cout << "Saved VFH signature to " << vfh_path << std::endl;

    // Load VFH signature of the query object
    if (!loadHist(vfh_path, histogram)) {
        pcl::console::print_error("Cannot load test file %s\n", vfh_path.c_str());
        return -1;
    }

    // Load training data
    loadData(model_directory, models);

    // Convert data into FLANN format
    flann::Matrix<float> data(new float[models.size() * models[0].second.size()],
        models.size(), models[0].second.size());

    for (size_t i = 0; i < data.rows; ++i)
        for (size_t j = 0; j < data.cols; ++j)
            data[i][j] = models[i].second[j];

    // Place data in FLANN K-d tree
    flann::Index<flann::ChiSquareDistance<float>> index(data, flann::LinearIndexParams());
    index.buildIndex();

    // Search for query object in the K-d tree
    nearestKSearch(index, histogram, k, k_indices, k_distances);

    // Print closest candidates on the console
    std::cout << "The closest " << k << " neighbors for " << input_name << " are: " << std::endl;
    for (int i = 0; i < k; ++i)
        pcl::console::print_info("    %d - %s (%d) with a distance of: %f\n",
                                 i, models.at(k_indices[0][i]).first.c_str(), k_indices[0][i], k_distances[0][i]);

    // Visualize the closest candidates on the screen
    visualize(argc, argv, k, thresh, models, k_indices, k_distances);

    // Join the thread before exiting
    img_thread.join();

    return 0;
}