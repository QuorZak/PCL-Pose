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
    pcl::PointCloud<pcl::PointXYZ> cloud_of_clusters; // Changed to object
    std::shared_ptr<std::vector<pcl::PointIndices>> cluster_indices(new std::vector<pcl::PointIndices>);
    std::vector<pcl::PointIndices> cluster_indices_copy;

    pcl::PointCloud<pcl::PointXYZ>::Ptr chosen_object_cloud(new pcl::PointCloud<pcl::PointXYZ>);

    std::mutex mtx;
    std::condition_variable condition_var;
    bool ready = false;

    // Start the streamDepthMap function in a separate thread
    cout << "Starting the point cloud streaming thread." << endl;
    std::thread img_thread(stream_point_cloud_show_depth_map, std::ref(pipe), std::ref(output_stream_cloud),
        std::ref(cluster_indices), std::ref(mtx), std::ref(condition_var), std::ref(ready));

    // While camera is initialising, load vfh model data
    load_vfh_model_data(model_directory, models);

    // Convert data into FLANN format
    flann::Matrix<float> data(new float[models.size() * models[0].second.size()],
        models.size(), models[0].second.size());

    for (size_t i = 0; i < data.rows; ++i)
        for (size_t j = 0; j < data.cols; ++j)
            data[i][j] = models[i].second[j];

    // Place data in FLANN K-d tree
    flann::Index<flann::ChiSquareDistance<float>> index(data, flann::LinearIndexParams());
    index.buildIndex();

    while (true) {
        // Wait for the first available output Realsense cloud
        {
            std::unique_lock<std::mutex> lock(mtx);
            condition_var.wait(lock, [&ready] { return ready; });
            // Once the lock is acquired, copy the output stream cloud to the pcl cloud
            output_stream_copy = *output_stream_cloud;
            cluster_indices_copy = *cluster_indices;
        }
        // Assign the copied point cloud to the cloud_of_clusters object
        cloud_of_clusters = output_stream_copy;

        pcl::PointCloud<pcl::PointXYZ>::Ptr best_cluster(new pcl::PointCloud<pcl::PointXYZ>);
        float best_distance = std::numeric_limits<float>::max();
        int best_index = -1;

        for (auto & [header, indices] : cluster_indices_copy) {
            pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_cluster(new pcl::PointCloud<pcl::PointXYZ>);
            for (const auto& idx : indices) {
                cloud_cluster->push_back(cloud_of_clusters[idx]);
            }

            // Estimate VFH signature
            pcl::PointCloud<pcl::VFHSignature308> signature;
            estimate_VFH(cloud_cluster, signature);

            // Search for the best match in the training models
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
            // Visualize the chosen model beside the best match model
            std::vector<std::string> files_to_show;
            files_to_show.push_back(output_directory + "/best_cluster.pcd");
            pcl::io::savePCDFileASCII(files_to_show.back(), *best_cluster);

            std::string best_model_path = models[best_index].first;
            files_to_show.push_back(best_model_path);

            showPointClouds(files_to_show);
        }

        std::cout << "Press any key to capture another point cloud, or Q to quit" << std::endl;
        char key;
        std::cin >> key;
        if (key == 'Q' || key == 'q') {
            break;
        }
    }

    return 0;
}