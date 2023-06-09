# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.24

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
CMAKE_COMMAND = /home/yunyi/.local/lib/python3.9/site-packages/cmake/data/bin/cmake

# The command to remove a file.
RM = /home/yunyi/.local/lib/python3.9/site-packages/cmake/data/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/yunyi/Desktop/VSCode_Proj/New-sycl-ml/SYCL-ML

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/yunyi/Desktop/VSCode_Proj/New-sycl-ml/SYCL-ML/build

# Include any dependencies generated for this target.
include example/CMakeFiles/run_gauss_classifier.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include example/CMakeFiles/run_gauss_classifier.dir/compiler_depend.make

# Include the progress variables for this target.
include example/CMakeFiles/run_gauss_classifier.dir/progress.make

# Include the compile flags for this target's objects.
include example/CMakeFiles/run_gauss_classifier.dir/flags.make

example/CMakeFiles/run_gauss_classifier.dir/src/mnist/run_gauss_classifier.cpp.o: example/CMakeFiles/run_gauss_classifier.dir/flags.make
example/CMakeFiles/run_gauss_classifier.dir/src/mnist/run_gauss_classifier.cpp.o: /home/yunyi/Desktop/VSCode_Proj/New-sycl-ml/SYCL-ML/example/src/mnist/run_gauss_classifier.cpp
example/CMakeFiles/run_gauss_classifier.dir/src/mnist/run_gauss_classifier.cpp.o: example/CMakeFiles/run_gauss_classifier.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/yunyi/Desktop/VSCode_Proj/New-sycl-ml/SYCL-ML/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object example/CMakeFiles/run_gauss_classifier.dir/src/mnist/run_gauss_classifier.cpp.o"
	cd /home/yunyi/Desktop/VSCode_Proj/New-sycl-ml/SYCL-ML/build/example && /home/yunyi/Desktop/VSCode_Proj/SYCL-ML-OneAPI/ComputeCpp-CE-1.2.0-Ubuntu-16.04-x86_64/bin/compute++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT example/CMakeFiles/run_gauss_classifier.dir/src/mnist/run_gauss_classifier.cpp.o -MF CMakeFiles/run_gauss_classifier.dir/src/mnist/run_gauss_classifier.cpp.o.d -o CMakeFiles/run_gauss_classifier.dir/src/mnist/run_gauss_classifier.cpp.o -c /home/yunyi/Desktop/VSCode_Proj/New-sycl-ml/SYCL-ML/example/src/mnist/run_gauss_classifier.cpp

example/CMakeFiles/run_gauss_classifier.dir/src/mnist/run_gauss_classifier.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/run_gauss_classifier.dir/src/mnist/run_gauss_classifier.cpp.i"
	cd /home/yunyi/Desktop/VSCode_Proj/New-sycl-ml/SYCL-ML/build/example && /home/yunyi/Desktop/VSCode_Proj/SYCL-ML-OneAPI/ComputeCpp-CE-1.2.0-Ubuntu-16.04-x86_64/bin/compute++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/yunyi/Desktop/VSCode_Proj/New-sycl-ml/SYCL-ML/example/src/mnist/run_gauss_classifier.cpp > CMakeFiles/run_gauss_classifier.dir/src/mnist/run_gauss_classifier.cpp.i

example/CMakeFiles/run_gauss_classifier.dir/src/mnist/run_gauss_classifier.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/run_gauss_classifier.dir/src/mnist/run_gauss_classifier.cpp.s"
	cd /home/yunyi/Desktop/VSCode_Proj/New-sycl-ml/SYCL-ML/build/example && /home/yunyi/Desktop/VSCode_Proj/SYCL-ML-OneAPI/ComputeCpp-CE-1.2.0-Ubuntu-16.04-x86_64/bin/compute++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/yunyi/Desktop/VSCode_Proj/New-sycl-ml/SYCL-ML/example/src/mnist/run_gauss_classifier.cpp -o CMakeFiles/run_gauss_classifier.dir/src/mnist/run_gauss_classifier.cpp.s

# Object files for target run_gauss_classifier
run_gauss_classifier_OBJECTS = \
"CMakeFiles/run_gauss_classifier.dir/src/mnist/run_gauss_classifier.cpp.o"

# External object files for target run_gauss_classifier
run_gauss_classifier_EXTERNAL_OBJECTS =

example/run_gauss_classifier: example/CMakeFiles/run_gauss_classifier.dir/src/mnist/run_gauss_classifier.cpp.o
example/run_gauss_classifier: example/CMakeFiles/run_gauss_classifier.dir/build.make
example/run_gauss_classifier: /home/yunyi/Desktop/VSCode_Proj/SYCL-ML-OneAPI/ComputeCpp-CE-1.2.0-Ubuntu-16.04-x86_64/lib/libComputeCpp.so
example/run_gauss_classifier: /usr/lib/x86_64-linux-gnu/libOpenCL.so
example/run_gauss_classifier: example/CMakeFiles/run_gauss_classifier.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/yunyi/Desktop/VSCode_Proj/New-sycl-ml/SYCL-ML/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable run_gauss_classifier"
	cd /home/yunyi/Desktop/VSCode_Proj/New-sycl-ml/SYCL-ML/build/example && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/run_gauss_classifier.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
example/CMakeFiles/run_gauss_classifier.dir/build: example/run_gauss_classifier
.PHONY : example/CMakeFiles/run_gauss_classifier.dir/build

example/CMakeFiles/run_gauss_classifier.dir/clean:
	cd /home/yunyi/Desktop/VSCode_Proj/New-sycl-ml/SYCL-ML/build/example && $(CMAKE_COMMAND) -P CMakeFiles/run_gauss_classifier.dir/cmake_clean.cmake
.PHONY : example/CMakeFiles/run_gauss_classifier.dir/clean

example/CMakeFiles/run_gauss_classifier.dir/depend:
	cd /home/yunyi/Desktop/VSCode_Proj/New-sycl-ml/SYCL-ML/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/yunyi/Desktop/VSCode_Proj/New-sycl-ml/SYCL-ML /home/yunyi/Desktop/VSCode_Proj/New-sycl-ml/SYCL-ML/example /home/yunyi/Desktop/VSCode_Proj/New-sycl-ml/SYCL-ML/build /home/yunyi/Desktop/VSCode_Proj/New-sycl-ml/SYCL-ML/build/example /home/yunyi/Desktop/VSCode_Proj/New-sycl-ml/SYCL-ML/build/example/CMakeFiles/run_gauss_classifier.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : example/CMakeFiles/run_gauss_classifier.dir/depend

