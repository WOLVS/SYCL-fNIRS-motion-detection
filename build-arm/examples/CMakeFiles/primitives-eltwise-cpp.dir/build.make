# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.25

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
CMAKE_COMMAND = /home/yunyi/.local/lib/python3.10/site-packages/cmake/data/bin/cmake

# The command to remove a file.
RM = /home/yunyi/.local/lib/python3.10/site-packages/cmake/data/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/yunyi/Desktop/zyy-workspace/oneDNN

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/yunyi/Desktop/zyy-workspace/oneDNN/build

# Include any dependencies generated for this target.
include examples/CMakeFiles/primitives-eltwise-cpp.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include examples/CMakeFiles/primitives-eltwise-cpp.dir/compiler_depend.make

# Include the progress variables for this target.
include examples/CMakeFiles/primitives-eltwise-cpp.dir/progress.make

# Include the compile flags for this target's objects.
include examples/CMakeFiles/primitives-eltwise-cpp.dir/flags.make

examples/CMakeFiles/primitives-eltwise-cpp.dir/primitives/eltwise.cpp.o: examples/CMakeFiles/primitives-eltwise-cpp.dir/flags.make
examples/CMakeFiles/primitives-eltwise-cpp.dir/primitives/eltwise.cpp.o: /home/yunyi/Desktop/zyy-workspace/oneDNN/examples/primitives/eltwise.cpp
examples/CMakeFiles/primitives-eltwise-cpp.dir/primitives/eltwise.cpp.o: examples/CMakeFiles/primitives-eltwise-cpp.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/yunyi/Desktop/zyy-workspace/oneDNN/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object examples/CMakeFiles/primitives-eltwise-cpp.dir/primitives/eltwise.cpp.o"
	cd /home/yunyi/Desktop/zyy-workspace/oneDNN/build/examples && /usr/bin/aarch64-linux-gnu-g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT examples/CMakeFiles/primitives-eltwise-cpp.dir/primitives/eltwise.cpp.o -MF CMakeFiles/primitives-eltwise-cpp.dir/primitives/eltwise.cpp.o.d -o CMakeFiles/primitives-eltwise-cpp.dir/primitives/eltwise.cpp.o -c /home/yunyi/Desktop/zyy-workspace/oneDNN/examples/primitives/eltwise.cpp

examples/CMakeFiles/primitives-eltwise-cpp.dir/primitives/eltwise.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/primitives-eltwise-cpp.dir/primitives/eltwise.cpp.i"
	cd /home/yunyi/Desktop/zyy-workspace/oneDNN/build/examples && /usr/bin/aarch64-linux-gnu-g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/yunyi/Desktop/zyy-workspace/oneDNN/examples/primitives/eltwise.cpp > CMakeFiles/primitives-eltwise-cpp.dir/primitives/eltwise.cpp.i

examples/CMakeFiles/primitives-eltwise-cpp.dir/primitives/eltwise.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/primitives-eltwise-cpp.dir/primitives/eltwise.cpp.s"
	cd /home/yunyi/Desktop/zyy-workspace/oneDNN/build/examples && /usr/bin/aarch64-linux-gnu-g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/yunyi/Desktop/zyy-workspace/oneDNN/examples/primitives/eltwise.cpp -o CMakeFiles/primitives-eltwise-cpp.dir/primitives/eltwise.cpp.s

# Object files for target primitives-eltwise-cpp
primitives__eltwise__cpp_OBJECTS = \
"CMakeFiles/primitives-eltwise-cpp.dir/primitives/eltwise.cpp.o"

# External object files for target primitives-eltwise-cpp
primitives__eltwise__cpp_EXTERNAL_OBJECTS =

examples/primitives-eltwise-cpp: examples/CMakeFiles/primitives-eltwise-cpp.dir/primitives/eltwise.cpp.o
examples/primitives-eltwise-cpp: examples/CMakeFiles/primitives-eltwise-cpp.dir/build.make
examples/primitives-eltwise-cpp: src/libdnnl.so.3.1
examples/primitives-eltwise-cpp: /usr/aarch64-linux-gnu/lib/libm.so
examples/primitives-eltwise-cpp: examples/CMakeFiles/primitives-eltwise-cpp.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/yunyi/Desktop/zyy-workspace/oneDNN/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable primitives-eltwise-cpp"
	cd /home/yunyi/Desktop/zyy-workspace/oneDNN/build/examples && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/primitives-eltwise-cpp.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
examples/CMakeFiles/primitives-eltwise-cpp.dir/build: examples/primitives-eltwise-cpp
.PHONY : examples/CMakeFiles/primitives-eltwise-cpp.dir/build

examples/CMakeFiles/primitives-eltwise-cpp.dir/clean:
	cd /home/yunyi/Desktop/zyy-workspace/oneDNN/build/examples && $(CMAKE_COMMAND) -P CMakeFiles/primitives-eltwise-cpp.dir/cmake_clean.cmake
.PHONY : examples/CMakeFiles/primitives-eltwise-cpp.dir/clean

examples/CMakeFiles/primitives-eltwise-cpp.dir/depend:
	cd /home/yunyi/Desktop/zyy-workspace/oneDNN/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/yunyi/Desktop/zyy-workspace/oneDNN /home/yunyi/Desktop/zyy-workspace/oneDNN/examples /home/yunyi/Desktop/zyy-workspace/oneDNN/build /home/yunyi/Desktop/zyy-workspace/oneDNN/build/examples /home/yunyi/Desktop/zyy-workspace/oneDNN/build/examples/CMakeFiles/primitives-eltwise-cpp.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : examples/CMakeFiles/primitives-eltwise-cpp.dir/depend

