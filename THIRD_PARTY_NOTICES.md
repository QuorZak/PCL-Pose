# Third-party notices

The code in this repository is licensed under the PolyForm Noncommercial License 1.0.0 (see `LICENSE`),
except for the parts listed below, which keep their original terms.

## Point Cloud Library (PCL) tutorial code

The VFH helper functions in `src/pose_estimation.h` (the `vfh_model` type, the VFH feature estimation,
the histogram and model loading, and the nearest-neighbour search) were adapted from the PCL tutorial
[Cluster Recognition and 6DOF Pose Estimation using VFH descriptors](https://pcl.readthedocs.io/projects/tutorials/en/master/vfh_recognition.html)
(`build_tree.cpp` and `nearest_neighbors.cpp`), through the work described below. PCL is distributed under the
BSD license, and its notice must be kept with those parts:

```
Software License Agreement (BSD License)

Point Cloud Library (PCL) - www.pointclouds.org
Copyright (c) 2009-2012, Willow Garage, Inc.
Copyright (c) 2012-, Open Perception, Inc.
Copyright (c) XXX, respective authors.

All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions
are met:

 * Redistributions of source code must retain the above copyright
   notice, this list of conditions and the following disclaimer.
 * Redistributions in binary form must reproduce the above
   copyright notice, this list of conditions and the following
   disclaimer in the documentation and/or other materials provided
   with the distribution.
 * Neither the name of the copyright holder(s) nor the names of its
   contributors may be used to endorse or promote products derived
   from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
"AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.
```

## dkebude/PCL-object-recognition

This project started from the work of Doğancan Kebude,
[dkebude/PCL-object-recognition](https://github.com/dkebude/PCL-object-recognition) (2017), which adapts the PCL
VFH tutorial into a small object recognition program. That repository does not state a licence. It is credited here
as the starting point, and none of its files are redistributed in this repository.
