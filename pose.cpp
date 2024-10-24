#include <pose.h>

int
main (int argc, char** argv)
{
	std::string inputname = "capture.pcd";
	std::string model_dir = "data";
	int k = 6;
	double thresh = DBL_MAX;

	bool cam = NULL;
	std::vector<vfh_model> models;	
  	flann::Matrix<int> k_indices;
  	flann::Matrix<float> k_distances;
  	
    	pcl::PointCloud<pcl::Normal> normals;
	pcl::PointCloud<pcl::VFHSignature308> vfh_descriptors;
	
	// Parse console inputs
	pcl::console::parse_argument (argc, argv, "-i", inputname);
	pcl::console::parse_argument (argc, argv, "-m", model_dir);
	pcl::console::parse_argument (argc, argv, "-k", k);
	pcl::console::parse_argument (argc, argv, "-t", thresh);
	
	rs2::pipeline pipeline;
	rs2::pipeline_profile pipe_stream;
    	rs2::config config;
    	rs2::frameset frames;
    	rs2::frame depth;
    	rs2::pointcloud rs_cloud;
	rs2::points points;
	
	try {
		// Start the Realsense camera
		config.enable_stream(RS2_STREAM_DEPTH, 640, 480, RS2_FORMAT_Z16, 15);
		pipe_stream = pipeline.start(config);
		std::cout << "Camera initialized successfully." << std::endl;
	    	// Camera warmup - dropping several first frames to let auto-exposure stabilize
		for(int i = 0; i < 100; i++)
		{
			// Wait for all configured streams to produce a frame
			frames = pipeline.wait_for_frames();
		}
		// Capture a depth frame from the Realsense camera
		frames = pipeline.wait_for_frames();
		depth = frames.get_depth_frame();
		if (!depth) {
			std::cerr << "No frames received." << std::endl;
			return(-1); 
		}
	        // Intrinsics for depth scaling
		rs2_intrinsics intr = depth.get_profile().as<rs2::video_stream_profile>().get_intrinsics();
	} catch (const rs2::error& e) {
		std::cerr << "RealSense error: " << e.what() << std::endl;
		return(-1);
	}
	pipeline.stop();
	
	// Generate the point cloud from the depth frame
	points = rs_cloud.calculate(depth);

	// Convert to PCL format
	pcl::PointCloud<pcl::PointXYZ>::Ptr pcl_cloud(new pcl::PointCloud<pcl::PointXYZ>);
	auto vertices = points.get_vertices(); // Extract the 3D vertices

	pcl_cloud->width = points.size();
	pcl_cloud->height = 1; // Unordered point cloud
	pcl_cloud->is_dense = false;
	pcl_cloud->points.resize(points.size());

	for (size_t i = 0; i < points.size(); ++i) {
	    pcl_cloud->points[i].x = vertices[i].x;
	    pcl_cloud->points[i].y = vertices[i].y;
	    pcl_cloud->points[i].z = vertices[i].z;
	}

	// Finished getting point cloud, now getting VFH
	std::cout << "Finished getting point cloud, now getting VFH." << std::endl;
  	vfh_model histogram;
  	std::string str = inputname;
	// Estimate signature and save it in a file
	pcl::PointCloud <pcl::VFHSignature308> signature;
	estimate_VFH(pcl_cloud, signature);
	str.replace(str.find(".pcd"), 4, "_vfh.pcd");
	pcl::io::savePCDFile (inputname, signature);
	// Load VFH signature of the query object
    	if(!loadHist (str, histogram))
  	{
  	  pcl::console::print_error ("Cannot load test file %s\n", inputname);
  	  return (-1);
  	}
	
	// Load training data
	loadData(model_dir, models);

  	// Convert data into FLANN format
  	flann::Matrix<float> data (new float[models.size () * models[0].second.size ()], models.size (), models[0].second.size ());
	
  	for (size_t i = 0; i < data.rows; ++i)
  	  for (size_t j = 0; j < data.cols; ++j)
  	    data[i][j] = models[i].second[j];

	// Place data in FLANN K-d tree
	flann::Index<flann::ChiSquareDistance<float> > index (data, flann::LinearIndexParams ());
	index.buildIndex ();
	
	// Search for query object in the K-d tree
	nearestKSearch (index, histogram, k, k_indices, k_distances);

	// Print closest candidates on the console
	std::cout << "The closest " << k << " neighbors for " << inputname << " are: " << std::endl;
  	for (int i = 0; i < k; ++i)
    	pcl::console::print_info ("    %d - %s (%d) with a distance of: %f\n", 
        i, models.at (k_indices[0][i]).first.c_str (), k_indices[0][i], k_distances[0][i]);

	// Visualize closest candidates on the screen
	visualize(argc, argv, k, thresh, models, k_indices, k_distances);

	return 0;
}
