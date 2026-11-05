# PCL-Pose
A series of tools and functions for object recognition and 6DOF pose estimation using [PCL](http://pointclouds.org/), an Intel RealSense D455 depth camera and a robot arm. It was developed for my Master's research report, [Improving Robotic Perception by Refining 6D Object Pose Estimation using Push Based Manipulation](docs/PCL-Pose_Research_Report_2024-11-11.pdf) (University of Auckland, 2024), where the pose and approach point it calculates are used to push objects with a UR5e cobot.

The approach is PCL's cluster recognition with VFH (viewpoint feature histogram) descriptors. The VFH documentation can be found in the PCL [tutorial](https://pcl.readthedocs.io/projects/tutorials/en/master/vfh_recognition.html). The starting point for this project was the official PCL documentation as well as the [previous work](https://github.com/dkebude/PCL-object-recognition) of Doğancan Kebude.

## How it works
```
RealSense D455 depth frames
  |  depth filters (threshold, temporal), crop to the centre of the image            camera
  v
point cloud
  |  voxel grid down-sampling (leaf size), RANSAC plane removal (floor, table)       segmentation
  |  Euclidean clustering: one cluster per object in the scene
  v
VFH signature of each cluster
  |  nearest-neighbour search (FLANN, chi-square distance) against lab_data/         vfh
  v
best matching object in the scene
  |  target band on the object, closest point, camera -> robot TCP transform,        manual_measure
  |  camera tilt (RX) from the accelerometer
  v
x, y, z (mm) and RX to move the robot to the object
```

## Hardware
- Intel RealSense D455, run at 848x480 and 15 fps, with the accelerometer used to find which way is up
- Universal Robots UR5e cobot. The program prints the move to make with respect to the tool (TCP) frame
- An end-effector baseplate for the camera (CAD files in `hardware/`)

## Results
From the report (the final tests used a leaf size of 4 mm):
- The pose estimates were within millimetres of the actual position: typically within 5 mm on the x-axis and 2 mm on the z-axis, with the depth (z) readings off by 3 to 4 mm at most.
- The x-axis varied the most, mainly because of the leaf size, which sets the spacing of the points.
- The robot TCP could move into a position to push the object using the vector from the pose estimate.

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
cmake ..
make -j$(nproc)
```
Run the programs from inside the `build` folder, because the default data paths are relative (`../lab_data/...`). The build type defaults to Release, because PCL and Eigen are very slow without optimisation. The experimental `pose` program is not part of the default build: use `make pose`.

## Repository contents
- `include/` and `src/` - The pose library (`pose_core`), shared by the programs, described below.
- `apps/` - The programs, described below.
- `config/` - YAML parameter files (`default.yaml` has every key with its default value).
- `lab_data/` - The object library: point clouds (`<object>_NNNN.pcd`) with their paired VFH signatures (`<object>_NNNN_vfh.pcd`), one folder per object. See `lab_data/LIBRARY_NOTES.md` for what is in it.
- `f_markers/` - ArUco marker images created by `generate_markers`.
- `docs/` - The research report.
- `hardware/` - CAD files (Inventor `.ipt` and `.stl`) for the D455 end-effector baseplate.

The library is split by job. `include/pose_estimation.h` pulls in all of it (and the third-party headers it uses), so a program only needs that one include:
- `include/pcl_pose/parameters.h` - Almost every possible parameter or global variable is declared here, so if you want to adjust the parameters for cluster extraction, or filtering for example, you are likely to find it in this file.
- `config.h` / `src/config.cpp` - Loading the parameters from a YAML file and the `--config` flag.
- `camera.h` / `src/camera.cpp` - RealSense depth filters and the conversion of a depth frame to a point cloud.
- `segmentation.h` / `src/segmentation.cpp` - Plane removal and cluster extraction, plus the thread function that does it continuously.
- `vfh.h` / `src/vfh.cpp` - VFH signatures, loading the library and the nearest-neighbour search.
- `pose.h` / `src/pose.cpp` - The pose manager, ICP, gravity alignment and the coordinate system overlay.
- `library.h` / `src/library.cpp` - Finding and centring the clouds of the object library.
- `viewer.h`, `files.h` - Point cloud viewing and listing files.

The programs in the `apps` folder:
- `manual_measure.cpp` (`manual_measure`) - This file measures the objects in the scene, finds the object that is most similar to the known library of point clouds and calculates the parameters for a robot arm to grasp the object.
- `cluster_extraction.cpp` (`extraction`) - Interactive tool for building the library. You enter the angle the object is facing, it captures until it finds exactly one cluster, places the coordinate system at the centre of the object, aligns it with gravity using the camera's accelerometer, and saves the cloud and its VFH into `lab_data/<object>/`. You review the result and keep or discard it.
- `cluster_extraction_test.cpp` (`extraction_test`) - Takes a single capture and saves every cluster it finds, with its VFH, into `lab_data/test/`. The clouds are saved as seen by the camera (no re-orientation), so move the ones you want into the library folder by hand, and keep in mind that they are not centred like the ones saved by `extraction`.
- `scratch_tools.cpp` (`scratch_tools`) - A scratch file for trying out the functions in the library (viewing saved clouds, streaming the depth map, matching a capture against the library). It is not an automated test suite.
- `main_marker.cpp` (`marker`) and `marker_pose_estimation.h` - Detect ArUco markers in the colour stream and print their pose. Useful for testing the pose, camera calibration, and distances.
- `generate_markers.cpp` (`generate_markers`) - Writes the ArUco marker images into `f_markers/`.
- `rebuild_vfh.cpp` (`rebuild_vfh`) - Checks that every cloud in the object library is centred on its centroid, and with `--apply` centres them and recomputes the paired VFH signatures. No camera needed. See `lab_data/LIBRARY_NOTES.md`.

Experimental, and not in the default build:
- `main.cpp` (`pose`, use `make pose`) - This file was intended to be the main file for the project. It attempts to find the 6D pose of the object by aligning the matching library cloud to the matched cluster with ICP, and draws the coordinate system on the colour image. The library is supposed to contain point clouds with the Z axis pointing forward, so the object must be rotated to match the library. Ensure you have a reliable library including defined x,y,z axis with respect to object pose.

## Configuration
The tunable values (camera stream, cloud filtering, ICP, the camera-to-TCP offset, the library location...) are described in `include/pcl_pose/parameters.h` and can be changed without recompiling from a YAML file. `config/default.yaml` lists every key with its default value, which suits the setup in my lab. Many of them are hard-coded to match the equipment I have available, so you will need to adjust them to match your own setup: copy the file, edit it, and run a program with `--config <your file>`. When no `--config` is given, a program started from the `build` folder loads `../config/default.yaml` if it exists, and keys that are missing keep their default value.

The programs also take command-line flags instead of editing the code (`--object`, `-m`, `-k`, `--home`, `--average`, `--calibrate`, `--debug-clusters`, `--debug-target`). Each program lists its own flags in the comment above `main()`.

## Setting up your own library
To set up your own library of point clouds, use `cluster_extraction.cpp` (the `extraction` program). Every cloud in the library should be centred on its centroid (see `lab_data/LIBRARY_NOTES.md`; `rebuild_vfh` checks this). Run it with `--object <name>`, and for each capture type the angle the object is facing. It saves each cluster and its paired VFH into `lab_data/<name>/`. Creating a folder for each object is recommended.

## Running the manual measure
Make sure you have very accurately set the parameters in the config file (see Configuration) to match your setup. You can then run the `manual_measure` program, for example `./manual_measure --home elevated`. It will measure the objects in the scene, find the object that is most similar to the known library of point clouds, and calculate the parameters for a robot arm to grasp the object. It is important that you point the program to the correct library location (`model_directory` in the config file, or `-m <folder>`) as it attempts to find the object in the scene that best matches a known object. You should see the output in the terminal.

## Known limitations
From the report:
- Reflective, dark, or harshly lit objects are hard to identify.
- There is a practical minimum leaf size: below it, noise makes extracting the object very difficult without a better cluster extraction.
- The minimum detection range is about 355 mm from the camera. Shallow camera angles add noise on perpendicular surfaces, and there is a slight bias towards objects in front of, or to the left of, the left-hand imager.

In the code:
- The library is a mix of centred and uncentred clouds, and the live programs do not centre the scene clusters, so the matching convention needs settling (`lab_data/LIBRARY_NOTES.md`).
- The averaging (`--average`) and the calibration adjustment (`--calibrate`) have not been tried on the robot.
- `pose` is experimental, and it reads colour frames from the same pipeline as its processing thread, so each only gets some of the framesets.
- There are no automated tests, and it has only been built on Ubuntu with a RealSense camera attached.

## Future work
From the report:
- Average the target point over several samples to reduce the variation.
- Add a ROS wrapper so the code can send the move to the UR5e, and keep `manual_measure` running to wait for the user's input.
- Add touch sensors on the end effector for feedback while the arm moves, using the pose manager for the sensor fusion.
- Refine the cluster extraction, and set the coordinate system of the pose in a more robust way than the centroid.
- Calculate a better approach vector that uses the 6D pose more.
- Train more objects, add a second RealSense camera, and test in cluttered scenes and different lighting.

## Licence
Free for noncommercial use, including personal use, research and education, under the [PolyForm Noncommercial License 1.0.0](LICENSE). For commercial use, please contact me for permission at zakquor@gmail.com.

Parts of the code were adapted from PCL's BSD-licensed tutorial code and keep their original notice. See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

### Acknowledgements
A special thank you to my supervisor and fellow researchers in the Mechatronics Department at the University of Auckland for their guidance and support. This project was developed as part of my Master's studies in Robotics and Automation Engineering.

Thank you to all the researchers who have shared any of their work via GitHub, in particular Doğancan Kebude's [PCL-object-recognition](https://github.com/dkebude/PCL-object-recognition). It has been an invaluable resource for learning and development. I would also like to thank the PCL community for their approachable and friendly documentation.
