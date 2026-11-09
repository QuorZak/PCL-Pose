# Object library notes

## DEPRECATED: the library in this folder is no longer useful
The clouds and signatures committed here (`spray_bottle_tall/` and `mustard_small/`) were captured with the old code,
and they do not match what the improved code computes. **A new library has to be built** with the `extraction`
program (see "Building the new library" below) before the matching results can be trusted again.

The old files are kept for now because the unit tests use them (they test loading and searching a library, not how
good the matches are) and as a record of the original work. Do not use them for measurements, and replace them once
the new library has been captured.

### Why the old library can not be used
The matching compares VFH signatures. Part of a VFH signature, and the direction of the normals it is built from,
depends on the viewpoint: where the camera was relative to the object. The scene clusters are matched as the camera
sees them, with the camera at the origin.

- The old `extraction` program centred each cloud on its centroid (and rotated it) *before* computing its signature,
  and left the viewpoint at the origin. The signature was therefore computed as if the camera were inside the object,
  so it does not match what the camera sees. This can not be repaired afterwards, because where the camera was is lost.
  This applies to `spray_bottle_tall_0001` to `_0008`.
- `spray_bottle_tall_0009` to `_0019` are raw camera-frame captures (centroid 0.35 to 0.48 m in front of the camera,
  viewpoint at the origin), so their signatures are consistent with the camera. But they were captured without the
  library convention (centred, y up, facing angle), and `_0018` and `_0019` were captured at a 1 mm leaf size (about
  10 000 points against 190 to 470 for the others), so their signatures are at a different density.
- `mustard_small` (12 clouds) has centroids at z of about -0.74 to -0.98 m, behind the camera, so it was transformed
  in a way that is not recorded, and its viewpoint can not be trusted either.

Keeping a mixed library would make the results depend on which captures happen to be closest, so the whole library is
retired and rebuilt with one convention.

## How the improved code handles the viewpoint
Every library cloud stores where the camera was, in the cloud's own coordinates, as the `VIEWPOINT` field of its
`.pcd` file (`sensor_origin_` in PCL). `estimate_VFH` uses it for the normals and the signature. Whenever a library
cloud is moved (centred, rotated), its viewpoint is moved with it (`transformCloudAndViewpoint` and `centreOnCentroid`
in `library.h`), so its signature stays exactly the one the camera sees. The unit tests in `tests/test_vfh.cpp` check
this. The library can then follow the research report's convention (centred, y up, facing angle) without spoiling the
matching.

## Building the new library
Each object has its own folder, and every capture is a pair of files:
- `<object>_NNNN.pcd` - the point cloud of the object, centred, with its viewpoint
- `<object>_NNNN_vfh.pcd` - its VFH signature (308 bins), which is what the matching uses

1. Move the old folders out of the way (or delete them, they stay in the git history), so new captures do not mix with
   the old ones. The tests read `lab_data/spray_bottle_tall/`, so update `tests/test_vfh_library.cpp` if that folder
   changes or goes away.
2. From the `build` folder, run `./extraction --object <name>` for each object, with the same leaf size (`leaf_size` in
   the config) for every capture, and type the facing angle for each capture.
3. Check the result with `./rebuild_vfh -m ../lab_data/<name>/`: every cloud should be centred and none should say
   `VIEWPOINT LOST`.
4. Check the matching offline with `capture_scene` and `replay` before using it on the robot.
5. Point `model_directory` in the config (or `-m`) at the new folder, and update the table below.

`rebuild_vfh --apply` can centre a cloud that was saved in the camera frame (such as one from `extraction_test`) and
recompute its signature, moving the viewpoint with it, so the signature does not change. It refuses clouds whose
viewpoint is lost.

## State of the library
| Object | Clouds | State |
|---|---|---|
| `spray_bottle_tall` | 19 | Deprecated (see above) |
| `mustard_small` | 12 | Deprecated (see above) |
