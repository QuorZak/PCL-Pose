#pragma once
#include <string>

namespace pcl_pose {

// Loads the parameter values from a YAML file into the variables in parameters.h. The key is the name of the variable
// (see config/default.yaml). Keys that are not in the file keep their current value.
// Returns false if the file could not be opened.
bool loadConfig(const std::string& path);

// For the start of a program: loads the file given with --config <file>, or if there is none, config/default.yaml
// (looked for next to the build folder) when it exists. Returns false, after printing why, if a file could not be read.
// Call it before starting any threads.
bool loadConfigFromArguments(int argc, char** argv);

}  // namespace pcl_pose
