#include <pose_estimation.h>

int main(int argc, char** argv) {
    std::string input_name = "capture.pcd";
    std::string model_directory = "../data";
    std::string output_directory = "../output";
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
    rs2::pointcloud rs_cloud;
    rs2::points points;

    rs2::pipeline pipe;
    rs2::config cfg;
    cv::Mat output_depth_map;
    cv::Mat depth_map;
    std::mutex mtx;
    std::condition_variable condition_var;
    bool ready = false;

    // Start the streamDepthMap function in a separate thread
    std::thread img_thread(streamDepthMap, std::ref(pipe),std::ref(cfg),
        std::ref(output_depth_map), std::ref(mtx), std::ref(condition_var), std::ref(ready));

    // Wait for the first available output_depth_map
    {
        std::unique_lock<std::mutex> lock(mtx);
        condition_var.wait(lock, [&ready] { return ready; });
    }
    // Copy the output_depth_map to depth_map so that we can use it in the rest of the program
    depth_map = output_depth_map.clone();

    // Convert the depth frame to a PCL point cloud
    pcl::PointCloud<pcl::PointXYZ>::Ptr pcl_cloud(new pcl::PointCloud<pcl::PointXYZ>);
    for (int y = 0; y < depth_map.rows; ++y) {
        for (int x = 0; x < depth_map.cols; ++x) {
            if (uint16_t depth_value = depth_map.at<uint16_t>(y, x); depth_value > 0) {
                pcl::PointXYZ point;
                point.x = static_cast<float>(x);
                point.y = static_cast<float>(y);
                point.z = static_cast<float>(depth_value) * 0.001f; // Convert from mm to meters
                pcl_cloud->points.push_back(point);
            }
        }
    }
    pcl_cloud->width = static_cast<uint32_t>(pcl_cloud->points.size());
    pcl_cloud->height = 1;
    pcl_cloud->is_dense = false;
    pcl_cloud->sensor_origin_.setZero();

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

    return 0;
}