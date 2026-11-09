# PCL-Pose

[![Build](https://github.com/QuorZak/PCL-Pose/actions/workflows/build.yml/badge.svg)](https://github.com/QuorZak/PCL-Pose/actions/workflows/build.yml)

A series of tools and functions for object recognition and 6DOF pose estimation using [PCL](http://pointclouds.org/), an Intel RealSense D455 depth camera and a robot arm. It was developed for my Master's research report, [Improving Robotic Perception by Refining 6D Object Pose Estimation using Push Based Manipulation](docs/PCL-Pose_Research_Report_2024-11-11.pdf) (University of Auckland, 2024), where the pose and approach point it calculates are used to push objects with a UR5e cobot.

The approach is PCL's cluster recognition with VFH (viewpoint feature histogram) descriptors. The VFH documentation can be found in the PCL [tutorial](https://pcl.readthedocs.io/projects/tutorials/en/master/vfh_recognition.html). The starting point for this project was the official PCL documentation as well as the [previous work](https://github.com/dkebude/PCL-object-recognition) of Doğancan Kebude.

## Highlights
- **Perception pipeline in C++17 with PCL:** depth filtering, voxel down-sampling, RANSAC plane removal, Euclidean clustering, VFH object recognition with a FLANN nearest-neighbour search, and ICP pose refinement.
- **Hardware integration:** an Intel RealSense D455 (depth and accelerometer) and a Universal Robots UR5e cobot, with the camera processing on its own thread.
- **Accuracy:** in the lab tests the pose estimates were typically within 5 mm on the x-axis and 2 mm on the z-axis.
- **Engineering:** a modular library, YAML configuration, offline replay of saved scenes without a camera, GoogleTest unit tests, and a Docker build that runs on GitHub Actions.
- **History:** started as my Master's research in 2024 ([v0.1](https://github.com/QuorZak/PCL-Pose/releases/tag/v0.1-masters-research)), and reworked in 2026 ([v0.2](https://github.com/QuorZak/PCL-Pose/releases/tag/v0.2-first-major-rework)).

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

### Building in Docker
The `Dockerfile` sets up the whole build environment (Debian 13 with OpenCV 4.10 and PCL 1.15, plus librealsense built from source) and compiles every program, including the experimental `pose`, and runs the unit tests, so it is a way to check that the code builds without installing the dependencies:
```bash
docker build -t pcl-pose .
```
It only compiles the code and runs the tests. Running the programs needs a RealSense camera attached to the host. The same build runs on GitHub Actions for every push (`.github/workflows/build.yml`).

## Repository contents
- `include/` and `src/` - The pose library (`pose_core`), shared by the programs, described below.
- `apps/` - The programs, described below.
- `config/` - YAML parameter files (`default.yaml` has every key with its default value).
- `lab_data/` - The object library: point clouds (`<object>_NNNN.pcd`) with their paired VFH signatures (`<object>_NNNN_vfh.pcd`), one folder per object. **The library committed there is deprecated and has to be rebuilt**: see `lab_data/LIBRARY_NOTES.md`.
- `f_markers/` - ArUco marker images created by `generate_markers`.
- `tests/` - The unit tests.
- `docs/` - The research report.
- `hardware/` - CAD files (Inventor `.ipt` and `.stl`) for the D455 end-effector baseplate.

The library is split by job. `include/pose_estimation.h` pulls in all of it (and the third-party headers it uses), so a program only needs that one include:
- `include/pcl_pose/parameters.h` - Almost every possible parameter or global variable is declared here, so if you want to adjust the parameters for cluster extraction, or filtering for example, you are likely to find it in this file.
- `config.h` / `src/config.cpp` - Loading the parameters from a YAML file and the `--config` flag.
- `camera.h` / `src/camera.cpp` - RealSense depth filters and the conversion of a depth frame to a point cloud.
- `segmentation.h` / `src/segmentation.cpp` - Plane removal and cluster extraction, plus the thread function that does it continuously.
- `vfh.h` / `src/vfh.cpp` - VFH signatures (computed from the cloud's viewpoint, where the camera was) and loading the library's signatures.
- `pose.h` / `src/pose.cpp` - The pose manager, ICP, gravity alignment and the coordinate system overlay.
- `library.h` / `src/library.cpp` - Finding the clouds of the object library, and moving or centring a cloud together with its viewpoint.
- `vfh_library.h` / `src/vfh_library.cpp` - `VfhLibrary`: the library signatures with the index for finding the closest ones.
- `target.h` / `src/target.cpp` - Choosing the point on the object that the robot aims for.
- `viewer.h`, `files.h` - Point cloud viewing and listing files.

The programs in the `apps` folder:
- `manual_measure.cpp` (`manual_measure`) - This file measures the objects in the scene, finds the object that is most similar to the known library of point clouds and calculates the parameters for a robot arm to grasp the object.
- `cluster_extraction.cpp` (`extraction`) - Interactive tool for building the library. You enter the angle the object is facing, it captures until it finds exactly one cluster, places the coordinate system at the centre of the object, aligns it with gravity using the camera's accelerometer, and saves the cloud (with its viewpoint, so its VFH is still the one the camera sees) and its VFH into `lab_data/<object>/`. You review the result and keep or discard it.
- `cluster_extraction_test.cpp` (`extraction_test`) - Takes a single capture and saves every cluster it finds, with its VFH, into `lab_data/test/`. The clouds are saved as seen by the camera (no re-orientation), so move the ones you want into the library folder by hand, and centre them with `rebuild_vfh --apply` to match the ones saved by `extraction`.
- `scratch_tools.cpp` (`scratch_tools`) - A scratch file for trying out the functions in the library (viewing saved clouds, streaming the depth map, matching a capture against the library). It is not an automated test suite.
- `main_marker.cpp` (`marker`) and `marker_pose_estimation.h` - Detect ArUco markers in the colour stream and print their pose. Useful for testing the pose, camera calibration, and distances.
- `generate_markers.cpp` (`generate_markers`) - Writes the ArUco marker images into `f_markers/`.
- `replay.cpp` (`replay`) - Runs the whole recognition pipeline on a saved scene, no camera needed. See "Replaying scenes without a camera".
- `capture_scene.cpp` (`capture_scene`) - Saves one scene from the camera for `replay`.
- `rebuild_vfh.cpp` (`rebuild_vfh`) - Checks every cloud in the object library: whether it is centred on its centroid, and whether its viewpoint is usable. With `--apply` it centres the usable clouds (moving the viewpoint with them) and recomputes the paired VFH signatures. No camera needed. See `lab_data/LIBRARY_NOTES.md`.

Experimental, and not in the default build:
- `main.cpp` (`pose`, use `make pose`) - This file was intended to be the main file for the project. It attempts to find the 6D pose of the object by aligning the matching library cloud to the matched cluster with ICP (starting from the library cloud moved onto the cluster), and draws the coordinate system on the colour image. The library is supposed to contain point clouds with the Z axis pointing forward, so the object must be rotated to match the library. Ensure you have a reliable library including defined x,y,z axis with respect to object pose.

## Tests
`tests/` has [GoogleTest](https://github.com/google/googletest) unit tests for the parts that need no camera: the image crop, loading the config (including that `config/default.yaml` matches the defaults in the code), the file and library helpers (including moving a cloud with its viewpoint), that a VFH signature is unchanged when a cloud is moved or centred with its viewpoint, the pose manager and the gravity alignment, the target search (including coordinates that are all negative), matching against the VFH library committed in `lab_data/`, and segmenting a synthetic scene (a table with a can standing on it).
```bash
sudo apt install libgtest-dev
cmake -S . -B build && cmake --build build
ctest --test-dir build --output-on-failure
```
They are built by default when GoogleTest is installed (`-DPOSE_BUILD_TESTS=OFF` to skip them), and the Docker build and GitHub Actions run them. The camera code, the live threads and the robot interaction are not covered.

## Configuration
The tunable values (camera stream, cloud filtering, ICP, the camera-to-TCP offset, the library location...) are described in `include/pcl_pose/parameters.h` and can be changed without recompiling from a YAML file. `config/default.yaml` lists every key with its default value, which suits the setup in my lab. Many of them are hard-coded to match the equipment I have available, so you will need to adjust them to match your own setup: copy the file, edit it, and run a program with `--config <your file>`. When no `--config` is given, a program started from the `build` folder loads `../config/default.yaml` if it exists, and keys that are missing keep their default value.

The programs also take command-line flags instead of editing the code (`--object`, `-m`, `-k`, `--home`, `--average`, `--calibrate`, `--debug-clusters`, `--debug-target`). Each program lists its own flags in the comment above `main()`.

## Setting up your own library
The library committed in `lab_data/` is deprecated (it was captured with the old code, see `lab_data/LIBRARY_NOTES.md`), so a new one has to be built before the matching can be trusted. To set up a library of point clouds, use `cluster_extraction.cpp` (the `extraction` program). Run it with `--object <name>`, and for each capture type the angle the object is facing. It saves each cluster, centred and with its viewpoint, and its paired VFH into `lab_data/<name>/`. Creating a folder for each object is recommended, and use the same leaf size for every capture. `rebuild_vfh -m <folder>` checks the result. The steps are in `lab_data/LIBRARY_NOTES.md`.

## Replaying scenes without a camera
`replay` runs the whole recognition pipeline on a saved scene instead of the camera: it segments the scene into clusters, matches each cluster's VFH against the object library, and works out the point to aim for with the best match, using the same code and parameters as the live programs. So the matching, the parameters and the library can be worked on (and compared) without the camera plugged in or the robot set up.

To get a scene, capture one with the camera while it looks at the objects, then replay it as often as you like:
```bash
./capture_scene --name table      # saves ../scenes/table_0001.pcd
./replay ../scenes/table_0001.pcd
./replay ../scenes/table_0001.pcd -k 5 -m ../lab_data/spray_bottle_tall --config my_params.yaml
./replay ../scenes/table_0001.pcd --expect spray_bottle   # exit status 3 unless the best match has that in its name
```
Scenes are several MB each, so `/scenes/` is in `.gitignore`.

## Running the manual measure
Make sure you have very accurately set the parameters in the config file (see Configuration) to match your setup. You can then run the `manual_measure` program, for example `./manual_measure --home elevated`. It will measure the objects in the scene, find the object that is most similar to the known library of point clouds, and calculate the parameters for a robot arm to grasp the object. It is important that you point the program to the correct library location (`model_directory` in the config file, or `-m <folder>`) as it attempts to find the object in the scene that best matches a known object. When the best match is a spray bottle (its library file name contains `spray_bottle`), the collision offset `spray_collision_offset` is applied to the z move automatically. You should see the output in the terminal.

## Known limitations
From the report:
- Reflective, dark, or harshly lit objects are hard to identify.
- There is a practical minimum leaf size: below it, noise makes extracting the object very difficult without a better cluster extraction.
- The minimum detection range is about 355 mm from the camera. Shallow camera angles add noise on perpendicular surfaces, and there is a slight bias towards objects in front of, or to the left of, the left-hand imager.

In the code:
- The committed library is deprecated and has to be captured again with the improved code (`lab_data/LIBRARY_NOTES.md`).
- The averaging (`--average`) and the calibration adjustment (`--calibrate`) have not been tried on the robot.
- `pose` is experimental, and it reads colour frames from the same pipeline as its processing thread, so each only gets some of the framesets.
- Only the parts that need no camera have unit tests. The camera code, the live threads and the robot interaction have only been tried by hand, on Ubuntu with a RealSense camera attached.

What still needs testing on the camera and the robot, and the other open tasks, are listed in [TODO.md](TODO.md).

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
