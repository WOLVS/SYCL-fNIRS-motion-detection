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
include tests/gtests/CMakeFiles/test_iface_primitive_cache.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include tests/gtests/CMakeFiles/test_iface_primitive_cache.dir/compiler_depend.make

# Include the progress variables for this target.
include tests/gtests/CMakeFiles/test_iface_primitive_cache.dir/progress.make

# Include the compile flags for this target's objects.
include tests/gtests/CMakeFiles/test_iface_primitive_cache.dir/flags.make

tests/gtests/CMakeFiles/test_iface_primitive_cache.dir/main.cpp.o: tests/gtests/CMakeFiles/test_iface_primitive_cache.dir/flags.make
tests/gtests/CMakeFiles/test_iface_primitive_cache.dir/main.cpp.o: /home/yunyi/Desktop/zyy-workspace/oneDNN/tests/gtests/main.cpp
tests/gtests/CMakeFiles/test_iface_primitive_cache.dir/main.cpp.o: tests/gtests/CMakeFiles/test_iface_primitive_cache.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/yunyi/Desktop/zyy-workspace/oneDNN/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object tests/gtests/CMakeFiles/test_iface_primitive_cache.dir/main.cpp.o"
	cd /home/yunyi/Desktop/zyy-workspace/oneDNN/build/tests/gtests && /usr/bin/aarch64-linux-gnu-g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT tests/gtests/CMakeFiles/test_iface_primitive_cache.dir/main.cpp.o -MF CMakeFiles/test_iface_primitive_cache.dir/main.cpp.o.d -o CMakeFiles/test_iface_primitive_cache.dir/main.cpp.o -c /home/yunyi/Desktop/zyy-workspace/oneDNN/tests/gtests/main.cpp

tests/gtests/CMakeFiles/test_iface_primitive_cache.dir/main.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/test_iface_primitive_cache.dir/main.cpp.i"
	cd /home/yunyi/Desktop/zyy-workspace/oneDNN/build/tests/gtests && /usr/bin/aarch64-linux-gnu-g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/yunyi/Desktop/zyy-workspace/oneDNN/tests/gtests/main.cpp > CMakeFiles/test_iface_primitive_cache.dir/main.cpp.i

tests/gtests/CMakeFiles/test_iface_primitive_cache.dir/main.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/test_iface_primitive_cache.dir/main.cpp.s"
	cd /home/yunyi/Desktop/zyy-workspace/oneDNN/build/tests/gtests && /usr/bin/aarch64-linux-gnu-g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/yunyi/Desktop/zyy-workspace/oneDNN/tests/gtests/main.cpp -o CMakeFiles/test_iface_primitive_cache.dir/main.cpp.s

tests/gtests/CMakeFiles/test_iface_primitive_cache.dir/__/test_thread.cpp.o: tests/gtests/CMakeFiles/test_iface_primitive_cache.dir/flags.make
tests/gtests/CMakeFiles/test_iface_primitive_cache.dir/__/test_thread.cpp.o: /home/yunyi/Desktop/zyy-workspace/oneDNN/tests/test_thread.cpp
tests/gtests/CMakeFiles/test_iface_primitive_cache.dir/__/test_thread.cpp.o: tests/gtests/CMakeFiles/test_iface_primitive_cache.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/yunyi/Desktop/zyy-workspace/oneDNN/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object tests/gtests/CMakeFiles/test_iface_primitive_cache.dir/__/test_thread.cpp.o"
	cd /home/yunyi/Desktop/zyy-workspace/oneDNN/build/tests/gtests && /usr/bin/aarch64-linux-gnu-g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT tests/gtests/CMakeFiles/test_iface_primitive_cache.dir/__/test_thread.cpp.o -MF CMakeFiles/test_iface_primitive_cache.dir/__/test_thread.cpp.o.d -o CMakeFiles/test_iface_primitive_cache.dir/__/test_thread.cpp.o -c /home/yunyi/Desktop/zyy-workspace/oneDNN/tests/test_thread.cpp

tests/gtests/CMakeFiles/test_iface_primitive_cache.dir/__/test_thread.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/test_iface_primitive_cache.dir/__/test_thread.cpp.i"
	cd /home/yunyi/Desktop/zyy-workspace/oneDNN/build/tests/gtests && /usr/bin/aarch64-linux-gnu-g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/yunyi/Desktop/zyy-workspace/oneDNN/tests/test_thread.cpp > CMakeFiles/test_iface_primitive_cache.dir/__/test_thread.cpp.i

tests/gtests/CMakeFiles/test_iface_primitive_cache.dir/__/test_thread.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/test_iface_primitive_cache.dir/__/test_thread.cpp.s"
	cd /home/yunyi/Desktop/zyy-workspace/oneDNN/build/tests/gtests && /usr/bin/aarch64-linux-gnu-g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/yunyi/Desktop/zyy-workspace/oneDNN/tests/test_thread.cpp -o CMakeFiles/test_iface_primitive_cache.dir/__/test_thread.cpp.s

tests/gtests/CMakeFiles/test_iface_primitive_cache.dir/test_iface_primitive_cache.cpp.o: tests/gtests/CMakeFiles/test_iface_primitive_cache.dir/flags.make
tests/gtests/CMakeFiles/test_iface_primitive_cache.dir/test_iface_primitive_cache.cpp.o: /home/yunyi/Desktop/zyy-workspace/oneDNN/tests/gtests/test_iface_primitive_cache.cpp
tests/gtests/CMakeFiles/test_iface_primitive_cache.dir/test_iface_primitive_cache.cpp.o: tests/gtests/CMakeFiles/test_iface_primitive_cache.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/yunyi/Desktop/zyy-workspace/oneDNN/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Building CXX object tests/gtests/CMakeFiles/test_iface_primitive_cache.dir/test_iface_primitive_cache.cpp.o"
	cd /home/yunyi/Desktop/zyy-workspace/oneDNN/build/tests/gtests && /usr/bin/aarch64-linux-gnu-g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT tests/gtests/CMakeFiles/test_iface_primitive_cache.dir/test_iface_primitive_cache.cpp.o -MF CMakeFiles/test_iface_primitive_cache.dir/test_iface_primitive_cache.cpp.o.d -o CMakeFiles/test_iface_primitive_cache.dir/test_iface_primitive_cache.cpp.o -c /home/yunyi/Desktop/zyy-workspace/oneDNN/tests/gtests/test_iface_primitive_cache.cpp

tests/gtests/CMakeFiles/test_iface_primitive_cache.dir/test_iface_primitive_cache.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/test_iface_primitive_cache.dir/test_iface_primitive_cache.cpp.i"
	cd /home/yunyi/Desktop/zyy-workspace/oneDNN/build/tests/gtests && /usr/bin/aarch64-linux-gnu-g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/yunyi/Desktop/zyy-workspace/oneDNN/tests/gtests/test_iface_primitive_cache.cpp > CMakeFiles/test_iface_primitive_cache.dir/test_iface_primitive_cache.cpp.i

tests/gtests/CMakeFiles/test_iface_primitive_cache.dir/test_iface_primitive_cache.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/test_iface_primitive_cache.dir/test_iface_primitive_cache.cpp.s"
	cd /home/yunyi/Desktop/zyy-workspace/oneDNN/build/tests/gtests && /usr/bin/aarch64-linux-gnu-g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/yunyi/Desktop/zyy-workspace/oneDNN/tests/gtests/test_iface_primitive_cache.cpp -o CMakeFiles/test_iface_primitive_cache.dir/test_iface_primitive_cache.cpp.s

# Object files for target test_iface_primitive_cache
test_iface_primitive_cache_OBJECTS = \
"CMakeFiles/test_iface_primitive_cache.dir/main.cpp.o" \
"CMakeFiles/test_iface_primitive_cache.dir/__/test_thread.cpp.o" \
"CMakeFiles/test_iface_primitive_cache.dir/test_iface_primitive_cache.cpp.o"

# External object files for target test_iface_primitive_cache
test_iface_primitive_cache_EXTERNAL_OBJECTS =

tests/gtests/test_iface_primitive_cache: tests/gtests/CMakeFiles/test_iface_primitive_cache.dir/main.cpp.o
tests/gtests/test_iface_primitive_cache: tests/gtests/CMakeFiles/test_iface_primitive_cache.dir/__/test_thread.cpp.o
tests/gtests/test_iface_primitive_cache: tests/gtests/CMakeFiles/test_iface_primitive_cache.dir/test_iface_primitive_cache.cpp.o
tests/gtests/test_iface_primitive_cache: tests/gtests/CMakeFiles/test_iface_primitive_cache.dir/build.make
tests/gtests/test_iface_primitive_cache: src/libdnnl.so.3.1
tests/gtests/test_iface_primitive_cache: tests/gtests/gtest/libdnnl_gtest.a
tests/gtests/test_iface_primitive_cache: tests/gtests/CMakeFiles/test_iface_primitive_cache.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/yunyi/Desktop/zyy-workspace/oneDNN/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Linking CXX executable test_iface_primitive_cache"
	cd /home/yunyi/Desktop/zyy-workspace/oneDNN/build/tests/gtests && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/test_iface_primitive_cache.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
tests/gtests/CMakeFiles/test_iface_primitive_cache.dir/build: tests/gtests/test_iface_primitive_cache
.PHONY : tests/gtests/CMakeFiles/test_iface_primitive_cache.dir/build

tests/gtests/CMakeFiles/test_iface_primitive_cache.dir/clean:
	cd /home/yunyi/Desktop/zyy-workspace/oneDNN/build/tests/gtests && $(CMAKE_COMMAND) -P CMakeFiles/test_iface_primitive_cache.dir/cmake_clean.cmake
.PHONY : tests/gtests/CMakeFiles/test_iface_primitive_cache.dir/clean

tests/gtests/CMakeFiles/test_iface_primitive_cache.dir/depend:
	cd /home/yunyi/Desktop/zyy-workspace/oneDNN/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/yunyi/Desktop/zyy-workspace/oneDNN /home/yunyi/Desktop/zyy-workspace/oneDNN/tests/gtests /home/yunyi/Desktop/zyy-workspace/oneDNN/build /home/yunyi/Desktop/zyy-workspace/oneDNN/build/tests/gtests /home/yunyi/Desktop/zyy-workspace/oneDNN/build/tests/gtests/CMakeFiles/test_iface_primitive_cache.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : tests/gtests/CMakeFiles/test_iface_primitive_cache.dir/depend

