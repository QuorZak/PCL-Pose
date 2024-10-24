# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.22

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/zak/Repos/PCL_Zak

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/zak/Repos/PCL_Zak/build

# Include any dependencies generated for this target.
include CMakeFiles/nearest_neighbors.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/nearest_neighbors.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/nearest_neighbors.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/nearest_neighbors.dir/flags.make

CMakeFiles/nearest_neighbors.dir/nearest_neighbors.cpp.o: CMakeFiles/nearest_neighbors.dir/flags.make
CMakeFiles/nearest_neighbors.dir/nearest_neighbors.cpp.o: ../nearest_neighbors.cpp
CMakeFiles/nearest_neighbors.dir/nearest_neighbors.cpp.o: CMakeFiles/nearest_neighbors.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/zak/Repos/PCL_Zak/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/nearest_neighbors.dir/nearest_neighbors.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/nearest_neighbors.dir/nearest_neighbors.cpp.o -MF CMakeFiles/nearest_neighbors.dir/nearest_neighbors.cpp.o.d -o CMakeFiles/nearest_neighbors.dir/nearest_neighbors.cpp.o -c /home/zak/Repos/PCL_Zak/nearest_neighbors.cpp

CMakeFiles/nearest_neighbors.dir/nearest_neighbors.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/nearest_neighbors.dir/nearest_neighbors.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/zak/Repos/PCL_Zak/nearest_neighbors.cpp > CMakeFiles/nearest_neighbors.dir/nearest_neighbors.cpp.i

CMakeFiles/nearest_neighbors.dir/nearest_neighbors.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/nearest_neighbors.dir/nearest_neighbors.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/zak/Repos/PCL_Zak/nearest_neighbors.cpp -o CMakeFiles/nearest_neighbors.dir/nearest_neighbors.cpp.s

# Object files for target nearest_neighbors
nearest_neighbors_OBJECTS = \
"CMakeFiles/nearest_neighbors.dir/nearest_neighbors.cpp.o"

# External object files for target nearest_neighbors
nearest_neighbors_EXTERNAL_OBJECTS =

nearest_neighbors: CMakeFiles/nearest_neighbors.dir/nearest_neighbors.cpp.o
nearest_neighbors: CMakeFiles/nearest_neighbors.dir/build.make
nearest_neighbors: /usr/lib/x86_64-linux-gnu/libpcl_apps.so
nearest_neighbors: /usr/lib/x86_64-linux-gnu/libpcl_outofcore.so
nearest_neighbors: /usr/lib/x86_64-linux-gnu/libpcl_people.so
nearest_neighbors: /usr/lib/libOpenNI.so
nearest_neighbors: /usr/lib/x86_64-linux-gnu/libusb-1.0.so
nearest_neighbors: /usr/lib/x86_64-linux-gnu/libOpenNI2.so
nearest_neighbors: /usr/lib/x86_64-linux-gnu/libusb-1.0.so
nearest_neighbors: /usr/lib/x86_64-linux-gnu/libflann_cpp.so
nearest_neighbors: /usr/lib/x86_64-linux-gnu/hdf5/serial/libhdf5.so
nearest_neighbors: /usr/lib/x86_64-linux-gnu/libcrypto.so
nearest_neighbors: /usr/lib/x86_64-linux-gnu/libcurl.so
nearest_neighbors: /usr/lib/x86_64-linux-gnu/libpthread.a
nearest_neighbors: /usr/lib/x86_64-linux-gnu/libsz.so
nearest_neighbors: /usr/lib/x86_64-linux-gnu/libz.so
nearest_neighbors: /usr/lib/x86_64-linux-gnu/libdl.a
nearest_neighbors: /usr/lib/x86_64-linux-gnu/libm.so
nearest_neighbors: /usr/lib/x86_64-linux-gnu/libpcl_surface.so
nearest_neighbors: /usr/lib/x86_64-linux-gnu/libpcl_keypoints.so
nearest_neighbors: /usr/lib/x86_64-linux-gnu/libpcl_tracking.so
nearest_neighbors: /usr/lib/x86_64-linux-gnu/libpcl_recognition.so
nearest_neighbors: /usr/lib/x86_64-linux-gnu/libpcl_registration.so
nearest_neighbors: /usr/lib/x86_64-linux-gnu/libpcl_stereo.so
nearest_neighbors: /usr/lib/x86_64-linux-gnu/libpcl_segmentation.so
nearest_neighbors: /usr/lib/x86_64-linux-gnu/libpcl_features.so
nearest_neighbors: /usr/lib/x86_64-linux-gnu/libpcl_filters.so
nearest_neighbors: /usr/lib/x86_64-linux-gnu/libpcl_sample_consensus.so
nearest_neighbors: /usr/lib/x86_64-linux-gnu/libpcl_ml.so
nearest_neighbors: /usr/lib/x86_64-linux-gnu/libpcl_visualization.so
nearest_neighbors: /usr/lib/x86_64-linux-gnu/libpcl_search.so
nearest_neighbors: /usr/lib/x86_64-linux-gnu/libpcl_kdtree.so
nearest_neighbors: /usr/lib/x86_64-linux-gnu/libpcl_io.so
nearest_neighbors: /usr/lib/x86_64-linux-gnu/libpcl_octree.so
nearest_neighbors: /usr/lib/x86_64-linux-gnu/libpng.so
nearest_neighbors: /usr/lib/libOpenNI.so
nearest_neighbors: /usr/lib/x86_64-linux-gnu/libusb-1.0.so
nearest_neighbors: /usr/lib/x86_64-linux-gnu/libOpenNI2.so
nearest_neighbors: /usr/lib/x86_64-linux-gnu/libvtkChartsCore-9.1.so.9.1.0
nearest_neighbors: /usr/lib/x86_64-linux-gnu/libvtkInteractionImage-9.1.so.9.1.0
nearest_neighbors: /usr/lib/x86_64-linux-gnu/libvtkIOGeometry-9.1.so.9.1.0
nearest_neighbors: /usr/lib/x86_64-linux-gnu/libjsoncpp.so
nearest_neighbors: /usr/lib/x86_64-linux-gnu/libvtkIOPLY-9.1.so.9.1.0
nearest_neighbors: /usr/lib/x86_64-linux-gnu/libvtkRenderingLOD-9.1.so.9.1.0
nearest_neighbors: /usr/lib/x86_64-linux-gnu/libvtkViewsContext2D-9.1.so.9.1.0
nearest_neighbors: /usr/lib/x86_64-linux-gnu/libvtkViewsCore-9.1.so.9.1.0
nearest_neighbors: /usr/lib/x86_64-linux-gnu/libvtkGUISupportQt-9.1.so.9.1.0
nearest_neighbors: /usr/lib/x86_64-linux-gnu/libvtkInteractionWidgets-9.1.so.9.1.0
nearest_neighbors: /usr/lib/x86_64-linux-gnu/libvtkFiltersModeling-9.1.so.9.1.0
nearest_neighbors: /usr/lib/x86_64-linux-gnu/libvtkInteractionStyle-9.1.so.9.1.0
nearest_neighbors: /usr/lib/x86_64-linux-gnu/libvtkFiltersExtraction-9.1.so.9.1.0
nearest_neighbors: /usr/lib/x86_64-linux-gnu/libvtkIOLegacy-9.1.so.9.1.0
nearest_neighbors: /usr/lib/x86_64-linux-gnu/libvtkIOCore-9.1.so.9.1.0
nearest_neighbors: /usr/lib/x86_64-linux-gnu/libvtkRenderingAnnotation-9.1.so.9.1.0
nearest_neighbors: /usr/lib/x86_64-linux-gnu/libvtkRenderingContext2D-9.1.so.9.1.0
nearest_neighbors: /usr/lib/x86_64-linux-gnu/libvtkRenderingFreeType-9.1.so.9.1.0
nearest_neighbors: /usr/lib/x86_64-linux-gnu/libfreetype.so
nearest_neighbors: /usr/lib/x86_64-linux-gnu/libvtkImagingSources-9.1.so.9.1.0
nearest_neighbors: /usr/lib/x86_64-linux-gnu/libvtkIOImage-9.1.so.9.1.0
nearest_neighbors: /usr/lib/x86_64-linux-gnu/libvtkImagingCore-9.1.so.9.1.0
nearest_neighbors: /usr/lib/x86_64-linux-gnu/libvtkRenderingOpenGL2-9.1.so.9.1.0
nearest_neighbors: /usr/lib/x86_64-linux-gnu/libvtkRenderingUI-9.1.so.9.1.0
nearest_neighbors: /usr/lib/x86_64-linux-gnu/libvtkRenderingCore-9.1.so.9.1.0
nearest_neighbors: /usr/lib/x86_64-linux-gnu/libvtkCommonColor-9.1.so.9.1.0
nearest_neighbors: /usr/lib/x86_64-linux-gnu/libvtkFiltersGeometry-9.1.so.9.1.0
nearest_neighbors: /usr/lib/x86_64-linux-gnu/libvtkFiltersSources-9.1.so.9.1.0
nearest_neighbors: /usr/lib/x86_64-linux-gnu/libvtkFiltersGeneral-9.1.so.9.1.0
nearest_neighbors: /usr/lib/x86_64-linux-gnu/libvtkCommonComputationalGeometry-9.1.so.9.1.0
nearest_neighbors: /usr/lib/x86_64-linux-gnu/libvtkFiltersCore-9.1.so.9.1.0
nearest_neighbors: /usr/lib/x86_64-linux-gnu/libvtkCommonExecutionModel-9.1.so.9.1.0
nearest_neighbors: /usr/lib/x86_64-linux-gnu/libvtkCommonDataModel-9.1.so.9.1.0
nearest_neighbors: /usr/lib/x86_64-linux-gnu/libvtkCommonMisc-9.1.so.9.1.0
nearest_neighbors: /usr/lib/x86_64-linux-gnu/libvtkCommonTransforms-9.1.so.9.1.0
nearest_neighbors: /usr/lib/x86_64-linux-gnu/libvtkCommonMath-9.1.so.9.1.0
nearest_neighbors: /usr/lib/x86_64-linux-gnu/libvtkkissfft-9.1.so.9.1.0
nearest_neighbors: /usr/lib/x86_64-linux-gnu/libGLEW.so
nearest_neighbors: /usr/lib/x86_64-linux-gnu/libX11.so
nearest_neighbors: /usr/lib/x86_64-linux-gnu/libQt5OpenGL.so.5.15.3
nearest_neighbors: /usr/lib/x86_64-linux-gnu/libQt5Widgets.so.5.15.3
nearest_neighbors: /usr/lib/x86_64-linux-gnu/libQt5Gui.so.5.15.3
nearest_neighbors: /usr/lib/x86_64-linux-gnu/libQt5Core.so.5.15.3
nearest_neighbors: /usr/lib/x86_64-linux-gnu/libvtkCommonCore-9.1.so.9.1.0
nearest_neighbors: /usr/lib/x86_64-linux-gnu/libtbb.so.12.5
nearest_neighbors: /usr/lib/x86_64-linux-gnu/libvtksys-9.1.so.9.1.0
nearest_neighbors: /usr/lib/x86_64-linux-gnu/libpcl_common.so
nearest_neighbors: /usr/lib/x86_64-linux-gnu/libboost_system.so.1.74.0
nearest_neighbors: /usr/lib/x86_64-linux-gnu/libboost_filesystem.so.1.74.0
nearest_neighbors: /usr/lib/x86_64-linux-gnu/libboost_date_time.so.1.74.0
nearest_neighbors: /usr/lib/x86_64-linux-gnu/libboost_iostreams.so.1.74.0
nearest_neighbors: /usr/lib/x86_64-linux-gnu/libboost_serialization.so.1.74.0
nearest_neighbors: /usr/lib/x86_64-linux-gnu/libqhull_r.so.8.0.2
nearest_neighbors: /usr/lib/x86_64-linux-gnu/libz.so
nearest_neighbors: CMakeFiles/nearest_neighbors.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/zak/Repos/PCL_Zak/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable nearest_neighbors"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/nearest_neighbors.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/nearest_neighbors.dir/build: nearest_neighbors
.PHONY : CMakeFiles/nearest_neighbors.dir/build

CMakeFiles/nearest_neighbors.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/nearest_neighbors.dir/cmake_clean.cmake
.PHONY : CMakeFiles/nearest_neighbors.dir/clean

CMakeFiles/nearest_neighbors.dir/depend:
	cd /home/zak/Repos/PCL_Zak/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/zak/Repos/PCL_Zak /home/zak/Repos/PCL_Zak /home/zak/Repos/PCL_Zak/build /home/zak/Repos/PCL_Zak/build /home/zak/Repos/PCL_Zak/build/CMakeFiles/nearest_neighbors.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/nearest_neighbors.dir/depend

