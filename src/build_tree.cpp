#include <pose_estimation.h>

#include <pcl/common/pcl_filesystem.h>
#include <pcl/console/print.h>
#include <fstream>

typedef std::pair<std::string, std::vector<float> > vfh_model;

/** \brief Loads an n-D histogram file as a VFH signature
  * \param path the input file name
  * \param vfh the resultant VFH model
  */

/** \brief Load a set of VFH features that will act as the model (training data)
  * \param argc the number of arguments (pass from main ())
  * \param argv the actual command line arguments (pass from main ())
  * \param extension the file extension containing the VFH features
  * \param models the resultant vector of histogram models
  */
void loadFeatureModels (const pcl_fs::path &base_dir, const std::string &extension,
                   std::vector<vfh_model> &models)
{
  if (!pcl_fs::exists (base_dir) && !pcl_fs::is_directory (base_dir))
    return;

  for (pcl_fs::directory_iterator it (base_dir); it != pcl_fs::directory_iterator (); ++it)
  {
    if (pcl_fs::is_directory (it->status ()))
    {
      std::stringstream ss;
      ss << it->path ();
      pcl::console::print_highlight ("Loading %s (%lu models loaded so far).\n", ss.str ().c_str (), (unsigned long)models.size ());
      loadFeatureModels (it->path (), extension, models);
    }
    if (pcl_fs::is_regular_file (it->status ()) && it->path ().extension ().string () == extension)
    {
      vfh_model m;
      if (load_vfh_histogram (boost::filesystem::path(base_dir / it->path().filename()), m))
        models.push_back (m);
    }
  }
}

int main(int argc, char** argv) {
  std::string extension(".pcd");
  transform(extension.begin(), extension.end(), extension.begin(), static_cast<int(*)(int)>(tolower));

  std::string kdtree_idx_file_name = "kdtree.idx";
  std::string training_data_h5_file_name = "training_data.h5";
  std::string training_data_list_file_name = "training_data.list";

  std::vector<vfh_model> models;

  // Hard-coded data directory
  std::string data_directory = model_directory;

  // Load the model histograms
  loadFeatureModels(data_directory, extension, models);
  pcl::console::print_highlight("Loaded %d VFH models. Creating training data %s/%s.\n",
      static_cast<int>(models.size()), training_data_h5_file_name.c_str(), training_data_list_file_name.c_str());

  // Convert data into FLANN format
  flann::Matrix<float> data(new float[models.size() * models[0].second.size()], models.size(), models[0].second.size());

  for (std::size_t i = 0; i < data.rows; ++i)
    for (std::size_t j = 0; j < data.cols; ++j)
      data[i][j] = models[i].second[j];

  // Save data to disk (list of models)
  flann::save_to_file(data, training_data_h5_file_name, "training_data");
  std::ofstream fs;
  fs.open(training_data_list_file_name.c_str());
  for (std::size_t i = 0; i < models.size(); ++i)
    fs << models[i].first << "\n";
  fs.close();

  // Build the tree index and save it to disk
  pcl::console::print_error("Building the kdtree index (%s) for %d elements...\n", kdtree_idx_file_name.c_str(), (int)data.rows);
  flann::Index<flann::ChiSquareDistance<float>> index(data, flann::LinearIndexParams());
  //flann::Index<flann::ChiSquareDistance<float>> index(data, flann::KDTreeIndexParams(4));
  index.buildIndex();
  index.save(kdtree_idx_file_name);
  delete[] data.ptr();

  return 0;
}
