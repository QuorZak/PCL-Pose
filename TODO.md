# TODO

What still needs testing, checking or doing. The code compiles and the unit tests pass in the Docker build
(GitHub Actions), but everything that needs the RealSense camera or the robot has not been tried since the
2026 rework of the code. Tick items off as you go.

## 1. Build on the lab machine
- [ ] Build on the lab PC (Ubuntu, with the camera's librealsense): `mkdir build && cd build && cmake .. && make -j$(nproc)`.
- [ ] Build the experimental program too: `make pose`.
- [ ] Run the unit tests there: `ctest --test-dir build --output-on-failure` (needs `libgtest-dev`).
- [ ] If CMake stops on `JsonCpp::JsonCpp` or another VTK target, install `libjsoncpp-dev` (the workaround in
      `CMakeLists.txt` uses it).

## 2. Rebuild the object library
The committed library is deprecated (see `lab_data/LIBRARY_NOTES.md` for why, and the full steps).
- [ ] Move the old `lab_data/spray_bottle_tall/` and `lab_data/mustard_small/` folders out of the way.
- [ ] Capture each object with `./extraction --object <name>`, all at the same `leaf_size`.
- [ ] Run `./rebuild_vfh -m ../lab_data/<name>/`: every cloud should be centred, and none should say `VIEWPOINT LOST`.
- [ ] Open one saved `.pcd` and check its `VIEWPOINT` line is not `0 0 0 ...` (it should be where the camera was).
- [ ] Set `model_directory` in `config/default.yaml` (and the default in `include/pcl_pose/parameters.h`, they must
      match or the config test fails) to the new folder.
- [ ] Point `tests/test_vfh_library.cpp` at the new library folder, and check the tests still pass.
- [ ] Update the "State of the library" table in `lab_data/LIBRARY_NOTES.md`, and remove the DEPRECATED section and
      the README's mentions of it once the old files are gone.

## 3. Check each program with the camera
### capture_scene and replay
- [ ] `./capture_scene --name table` saves `../scenes/table_NNNN.pcd`.
- [ ] `./replay ../scenes/table_NNNN.pcd` can load it (this used to fail: the saved cloud's size did not match).
- [ ] With the new library, `replay` picks the right object, and `--expect spray_bottle` gives exit status 0.
- [ ] Compare the match distances against the old library, to see how much the viewpoint fix changed the matching.

### extraction (building the library)
- [ ] It starts without "No accelerometer data from the camera".
- [ ] Tilt the camera between two captures: the saved clouds should still come out with y up.
- [ ] Keeping (`y`) and discarding (`n`) a capture both work.

### extraction_test
- [ ] It saves the clusters into `lab_data/test/`, and `./rebuild_vfh -m ../lab_data/test/ --apply` centres them.

### manual_measure (on the robot)
- [ ] It finds and matches the object, and prints the move.
- [ ] Spray bottle: "Spray bottle matched: applied the collision offset" is printed, and only for a spray bottle.
      Check that -57 mm (`spray_collision_offset`) is still right.
- [ ] RX: tilt the camera one way and then the other, and check the sign of the printed RX matches the robot's
      rotation convention.
- [ ] "Press Enter to continue" waits for Enter, and `q` then Enter quits cleanly (no hang).
- [ ] After a robot move, the next measurement is from after the move (it should never use a frame taken while the
      robot was moving).
- [ ] `--calibrate`: after typing the three errors, the next "press Enter" waits (it used to be skipped), the new
      calibration adjuster is printed, and typing a letter instead of a number is rejected without crashing.
- [ ] `--average`, `--debug-target` and `--home front` (no accelerometer) still work.
- [ ] Measure the accuracy again and compare it with the results in the report.

### pose (experimental)
- [ ] ICP now starts from the library cloud moved onto the cluster: check the pose converges.
- [ ] The axes drawn on the colour image line up with the object (the pose is now moved into the colour camera's
      frame before drawing).

### Quick checks of the unchanged programs
- [ ] `marker` and `generate_markers` still work.
- [ ] Segmentation now carries on when no plane is found: check a scene without a table in view does not give junk
      clusters.

## 4. Repository
- [ ] Decide whether Kebude's `details.pdf`, which is no longer in the repository, must also be erased from the git
      history (only needed if it should never have been public; it means rewriting and force-pushing `main`).
- [ ] Tag the next version (e.g. `v0.3`) once the hardware checks pass. `v0.1-masters-research` is the code as it was
      for the research report (2024), and `v0.2-first-major-rework` is the reworked code before any hardware testing.
- [ ] GitHub Actions warns that `actions/checkout@v4`, `docker/setup-buildx-action@v3` and `docker/build-push-action@v6`
      run on the deprecated Node.js 20: update them to their newer major versions in `.github/workflows/build.yml`.

## 5. Known issues still in the code
- [ ] `pose`: the processing thread and the main loop both read frames from the same pipeline, so each only gets some
      of them. Give the colour frames to the main loop through the processing thread, as `manual_measure` does with the
      accelerometer.
- [ ] A camera error (for example a frame timeout) in the processing thread of `manual_measure` or `pose` ends the
      program, because the exception is not caught there.
- [ ] Points with no depth reading are still in the camera cloud as (0,0,0). They collapse into one point and are not
      a problem for clustering, but they could be filtered out.

## 6. Later (from the report's future work)
See "Future work" in the README: a ROS wrapper to send the move to the UR5e, touch sensors with the pose manager,
better cluster extraction and coordinate system, a better approach vector, and more objects, cameras and scenes.
