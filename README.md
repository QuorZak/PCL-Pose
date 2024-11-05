# PCL-PoseEstimation
This project provides a series of tools and functions developed for object recognition and pose estimation using PCL.
## Dependencies
### Major
- [Ubuntu (Project created on v20.04 - Focal)](https://ubuntu.com/)
- [PCL](http://pointclouds.org/)
- [OpenCV](https://opencv.org/)
- [Realsense SDK](https://dev.intelrealsense.com/docs/compiling-librealsense-for-linux-ubuntu-guide)
### Minor
- [Eigen](http://eigen.tuxfamily.org/index.php?title=Main_Page)
- [Boost](https://www.boost.org/)
- [HDF5](https://www.hdfgroup.org/solutions/hdf5/)
- [FLANN](https://www.cs.ubc.ca/research/flann/)
- [VTK](https://vtk.org/)
- [CMake](https://cmake.org/)


This is a [PCL](http://pointclouds.org/) based object recognition and pose estimation project based on PCL's cluster recognition and 6DOF pose estimation via VFH (viewpoint feature histogram) descriptors. The documentation for VFH can be found here [tutorial](http://pointclouds.org/documentation/tutorials/vfh_recognition.php#vfh-recognition). The starting point for my project work included leveraging the official PCL documentation as well as a researcher's [previous work](https://github.com/dkebude/PCL-object-recognition).

## Repository contents
In the src folder, you can find the following files:
- `pose_estimation.h` - This file contains the majority of the functions used for object recognition and pose estimation. It is used by practically all the other files in the project. I've declared almost every possible parameter or global variable here, so if you want to adjust the parameters for cluster extraction, or filtering for example, you are likely to find it at the top of this file.
- `cluster_extraction_test.cpp` - This file extracts clusters from a point cloud. You can use this to create your own library of data (object clouds). Move extracted files from data/test to the desired data/xx folder.
- `manual_measure.cpp` - This file measures the objects in the scene, finds the object that is most similar to the known library of point clouds and calculates the parameters for a robot arm to grasp the object.
- `tests.cpp` - Contains tests for the functions in `pose_estimation.h`. It is used to ensure that the functions are working as expected. It can also be used to see the extracted point clouds from the `cluster_extraction_test.cpp` file.

Files that need work to be run reliably, or are not currently working:
- `cluster_extraction.cpp` - This file contains the code for extracting clusters from a point cloud in a loop. It's purpose was to quickly and efficiently extract clusters.
- `main.cpp` - This file was intended to be the main file for the project. It attempts to find the 6D pose of the object by comparing the point cloud to the library of point clouds. The library is supposed to contain point clouds with the Z axis pointing forward, so the object must be rotated to match the library. Ensure you have a reliable library including defined x,y,z axis with respect to object pose.

Marker files:
- `marker_pose_estimation.h` and `marker_tests.cpp` - These files were used to test the pose estimation of a marker. They use the ArUco library to detect physical markers placed in the scene. Useful for testing the pose, camera calibration, and distances.

## Setting up the project for your own configuration
You should review all the parameters defined in the `pose_estimation.h` file. Many of the parameters are hard-coded to match the setup in my lab, and the equipment I have available. You will need to adjust these parameters to match your own setup. The parameters are defined at the top of the file, and are well commented. You should be able to adjust the parameters to match your setup.

## Setting up your own library
To set up your own library of point clouds, you can use the `cluster_extraction_test.cpp` file. This file extracts clusters from a point cloud and saves them to a folder. It also saves the paired VFH. Find the correct file for the object(s) in the scene and move them to the desired data/xx folder. You can create a folder for each object if you desire (recommended).

## Running the manual measure
Make sure you have very accurately changed the hard-coded parameters in the `pose_estimation.h` file to match your setup. You can then run the `manual_measure.cpp` file. This file will measure the objects in the scene, find the object that is most similar to the known library of point clouds, and calculate the parameters for a robot arm to grasp the object. It is important that you point the function to the correct data library location as it attempts to find the object in the scene that best matches a known object. You should see the output in the terminal.

### Acknowledgements
A special thank you to my supervisor and fellow researchers in the Mechatronics Department at the University of Auckland for their guidance and support. This project was developed as part of my Master's studies in Robotics and Automation Engineering.

Thank you to all the researchers who have shared any of their work via GitHub. It has been an invaluable resource for learning and development. I would also like to thank the PCL community for their approachable and friendly documentation.