# PCL-Pose
A series of tools and functions for object recognition and 6DOF pose estimation using [PCL](http://pointclouds.org/), an Intel RealSense D455 depth camera and a robot arm. It was developed for my Master's research report, [Improving Robotic Perception by Refining 6D Object Pose Estimation using Push Based Manipulation](docs/PCL-Pose_Research_Report_2024-11-11.pdf) (University of Auckland, 2024), where the pose and approach point it calculates are used to push objects with a UR5e cobot.

The approach is PCL's cluster recognition with VFH (viewpoint feature histogram) descriptors: planes are removed from the depth image, the remaining points are clustered, each cluster's VFH is matched against a library of known objects, and the best match is turned into a target point for the arm. The VFH documentation can be found in the PCL [tutorial](https://pcl.readthedocs.io/projects/tutorials/en/master/vfh_recognition.html). The starting point for this project was the official PCL documentation as well as the [previous work](https://github.com/dkebude/PCL-object-recognition) of Doğancan Kebude.

## Dependencies
### Major
- [Ubuntu (Project created on v20.04 - Focal)](https://ubuntu.com/)
- [PCL](http://pointclouds.org/) (1.8 or newer)
- [OpenCV](https://opencv.org/) (4.7 or newer, for the `cv::aruco::ArucoDetector` API)
- [Realsense SDK](https://dev.intelrealsense.com/docs/compiling-librealsense-for-linux-ubuntu-guide)
### Minor
- [Eigen](http://eigen.tuxfamily.org/index.php?title=Main_Page)
- [Boost](https://www.boost.org/)
- [HDF5](https://www.hdfgroup.org/solutions/hdf5/)
- [FLANN](https://www.cs.ubc.ca/research/flann/)
- [VTK](https://vtk.org/)
- [CMake](https://cmake.org/)

## Building
```bash
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j$(nproc)
```
Run the programs from inside the `build` folder, because the default data paths are relative (`../lab_data/...`). A Release build matters: PCL and Eigen are very slow without optimisation.

## Repository contents
- `src/` - The source code, described below.
- `lab_data/` - The object library: point clouds (`<object>_NNNN.pcd`) with their paired VFH signatures (`<object>_NNNN_vfh.pcd`), one folder per object.
- `f_markers/` - ArUco marker images created by `generate_markers`.
- `docs/` - The research report.
- `hardware/` - CAD files (Inventor `.ipt` and `.stl`) for the D455 end-effector baseplate.

In the `src` folder you can find the following files:
- `pose_estimation.h` - This file contains the majority of the functions used for object recognition and pose estimation. It is used by practically all the other files in the project. I've declared almost every possible parameter or global variable here, so if you want to adjust the parameters for cluster extraction, or filtering for example, you are likely to find it at the top of this file.
- `manual_measure.cpp` (`manual_measure`) - This file measures the objects in the scene, finds the object that is most similar to the known library of point clouds and calculates the parameters for a robot arm to grasp the object.
- `cluster_extraction.cpp` (`extraction`) - Interactive tool for building the library. You enter the angle the object is facing, it captures until it finds exactly one cluster, places the coordinate system at the centre of the object, aligns it with gravity using the camera's accelerometer, and saves the cloud and its VFH into `lab_data/<object>/`. You review the result and keep or discard it.
- `cluster_extraction_test.cpp` (`extraction_test`) - Takes a single capture and saves every cluster it finds, with its VFH, into `lab_data/test/`. The clouds are saved as seen by the camera (no re-orientation), so move the ones you want into the library folder by hand, and keep in mind that they are not centred like the ones saved by `extraction`.
- `tests.cpp` (`tests`) - A scratch file for trying out the functions in `pose_estimation.h` (viewing saved clouds, streaming the depth map, matching a capture against the library). It is not an automated test suite.
- `main_marker.cpp` (`marker`) and `marker_pose_estimation.h` - Detect ArUco markers in the colour stream and print their pose. Useful for testing the pose, camera calibration, and distances.
- `generate_markers.cpp` (`generate_markers`) - Writes the ArUco marker images into `f_markers/`.

Files that need work to be run reliably, or are not currently working:
- `main.cpp` (`pose`) - This file was intended to be the main file for the project. It attempts to find the 6D pose of the object by comparing the point cloud to the library of point clouds. The library is supposed to contain point clouds with the Z axis pointing forward, so the object must be rotated to match the library. Ensure you have a reliable library including defined x,y,z axis with respect to object pose.

## Setting up the project for your own configuration
You should review all the parameters defined in the `pose_estimation.h` file. Many of the parameters are hard-coded to match the setup in my lab, and the equipment I have available. You will need to adjust these parameters to match your own setup. The parameters are defined at the top of the file, and are well commented. You should be able to adjust the parameters to match your setup.

## Setting up your own library
To set up your own library of point clouds, use `cluster_extraction.cpp` (the `extraction` program). Set `object_name` in the file, run it, and for each capture type the angle the object is facing. It saves each cluster and its paired VFH into `lab_data/<object_name>/`. Creating a folder for each object is recommended.

## Running the manual measure
Make sure you have very accurately changed the hard-coded parameters in the `pose_estimation.h` file to match your setup. You can then run the `manual_measure` program. It will measure the objects in the scene, find the object that is most similar to the known library of point clouds, and calculate the parameters for a robot arm to grasp the object. It is important that you point the program to the correct library location (`model_directory` in `pose_estimation.h`) as it attempts to find the object in the scene that best matches a known object. You should see the output in the terminal.

## Licence
Free for noncommercial use, including personal use, research and education, under the [PolyForm Noncommercial License 1.0.0](LICENSE). For commercial use, please contact me for permission at zakquor@gmail.com.

Parts of the code were adapted from PCL's BSD-licensed tutorial code and keep their original notice. See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

### Acknowledgements
A special thank you to my supervisor and fellow researchers in the Mechatronics Department at the University of Auckland for their guidance and support. This project was developed as part of my Master's studies in Robotics and Automation Engineering.

Thank you to all the researchers who have shared any of their work via GitHub, in particular Doğancan Kebude's [PCL-object-recognition](https://github.com/dkebude/PCL-object-recognition). It has been an invaluable resource for learning and development. I would also like to thank the PCL community for their approachable and friendly documentation.
