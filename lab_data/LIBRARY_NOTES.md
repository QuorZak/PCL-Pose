# Object library notes

## Layout
Each object has its own folder. Every capture is a pair of files:
- `<object>_NNNN.pcd` - the point cloud of the object
- `<object>_NNNN_vfh.pcd` - its VFH signature (308 bins), which is what the matching uses

## Convention
The research report places the coordinate system of every library cloud at the centroid of the object, with y up
(worked out from the camera's accelerometer) and the facing angle of the object set when it is captured.
The `extraction` program does this. The `extraction_test` program does not: it saves the clouds as the camera saw them.

## State of the library
Measured on the committed files (a cloud counts as centred if its centroid is within 1 mm of the origin):

| Object | Clouds | Centred | Not centred |
|---|---|---|---|
| `spray_bottle_tall` | 19 | 8 (`0001` to `0008`) | 11 (`0009` to `0019`): centroid between 0.35 and 0.48 m in front of the camera, so they are raw camera-frame captures |
| `mustard_small` | 12 | 0 | 12: centroid z of about -0.74 to -0.98 m |

Other things noticed:
- `spray_bottle_tall_0018` and `_0019` have about 10 000 points, against about 190 to 470 for most of the others. They were added
  in the commit that decreased the leaf size to 1 mm (the git history has the leaf size going to 1 mm and back to 4 mm on
  2024-11-05). The signature is computed at a different density, so these two are worth recapturing.
- The live programs (`manual_measure`, `pose`) search the whole `lab_data/spray_bottle_tall/` folder, so they match
  against this mix of centred and uncentred clouds. `mustard_small` is not used unless `model_directory` is changed.

## Why it matters
A VFH signature has a part that depends on the viewpoint (the camera at the origin), so the same object gives a
different signature when it is centred than when it is in the camera frame. The library should follow one convention.
Also check offline, before changing how the live programs prepare the scene clusters, what the match distances do:
the programs currently do not centre the scene clusters.

## The tool
`rebuild_vfh` (no camera needed), run from the `build` folder:
```bash
./rebuild_vfh            # lists every cloud and whether it is centred, changes nothing
./rebuild_vfh --apply    # centres the clouds and recomputes every _vfh.pcd so each pair matches
```
Use `-m <folder>` to point it at a different library folder.

## Status
Not applied yet: the files in this folder are still as captured. To finish, run `rebuild_vfh --apply` on a machine with
PCL, check the matching offline, commit the regenerated files, and record the date and what changed here.
