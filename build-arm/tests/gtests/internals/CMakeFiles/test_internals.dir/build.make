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
include tests/gtests/internals/CMakeFiles/test_internals.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include tests/gtests/internals/CMakeFiles/test_internals.dir/compiler_depend.make

# Include the progress variables for this target.
include tests/gtests/internals/CMakeFiles/test_internals.dir/progress.make

# Include the compile flags for this target's objects.
include tests/gtests/internals/CMakeFiles/test_internals.dir/flags.make

tests/gtests/internals/CMakeFiles/test_internals.dir/test_bcast_strategy.cpp.o: tests/gtests/internals/CMakeFiles/test_internals.dir/flags.make
tests/gtests/internals/CMakeFiles/test_internals.dir/test_bcast_strategy.cpp.o: /home/yunyi/Desktop/zyy-workspace/oneDNN/tests/gtests/internals/test_bcast_strategy.cpp
tests/gtests/internals/CMakeFiles/test_internals.dir/test_bcast_strategy.cpp.o: tests/gtests/internals/CMakeFiles/test_internals.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/yunyi/Desktop/zyy-workspace/oneDNN/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object tests/gtests/internals/CMakeFiles/test_internals.dir/test_bcast_strategy.cpp.o"
	cd /home/yunyi/Desktop/zyy-workspace/oneDNN/build/tests/gtests/internals && /usr/bin/aarch64-linux-gnu-g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT tests/gtests/internals/CMakeFiles/test_internals.dir/test_bcast_strategy.cpp.o -MF CMakeFiles/test_internals.dir/test_bcast_strategy.cpp.o.d -o CMakeFiles/test_internals.dir/test_bcast_strategy.cpp.o -c /home/yunyi/Desktop/zyy-workspace/oneDNN/tests/gtests/internals/test_bcast_strategy.cpp

tests/gtests/internals/CMakeFiles/test_internals.dir/test_bcast_strategy.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/test_internals.dir/test_bcast_strategy.cpp.i"
	cd /home/yunyi/Desktop/zyy-workspace/oneDNN/build/tests/gtests/internals && /usr/bin/aarch64-linux-gnu-g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/yunyi/Desktop/zyy-workspace/oneDNN/tests/gtests/internals/test_bcast_strategy.cpp > CMakeFiles/test_internals.dir/test_bcast_strategy.cpp.i

tests/gtests/internals/CMakeFiles/test_internals.dir/test_bcast_strategy.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/test_internals.dir/test_bcast_strategy.cpp.s"
	cd /home/yunyi/Desktop/zyy-workspace/oneDNN/build/tests/gtests/internals && /usr/bin/aarch64-linux-gnu-g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/yunyi/Desktop/zyy-workspace/oneDNN/tests/gtests/internals/test_bcast_strategy.cpp -o CMakeFiles/test_internals.dir/test_bcast_strategy.cpp.s

tests/gtests/internals/CMakeFiles/test_internals.dir/test_bfloat16.cpp.o: tests/gtests/internals/CMakeFiles/test_internals.dir/flags.make
tests/gtests/internals/CMakeFiles/test_internals.dir/test_bfloat16.cpp.o: /home/yunyi/Desktop/zyy-workspace/oneDNN/tests/gtests/internals/test_bfloat16.cpp
tests/gtests/internals/CMakeFiles/test_internals.dir/test_bfloat16.cpp.o: tests/gtests/internals/CMakeFiles/test_internals.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/yunyi/Desktop/zyy-workspace/oneDNN/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object tests/gtests/internals/CMakeFiles/test_internals.dir/test_bfloat16.cpp.o"
	cd /home/yunyi/Desktop/zyy-workspace/oneDNN/build/tests/gtests/internals && /usr/bin/aarch64-linux-gnu-g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT tests/gtests/internals/CMakeFiles/test_internals.dir/test_bfloat16.cpp.o -MF CMakeFiles/test_internals.dir/test_bfloat16.cpp.o.d -o CMakeFiles/test_internals.dir/test_bfloat16.cpp.o -c /home/yunyi/Desktop/zyy-workspace/oneDNN/tests/gtests/internals/test_bfloat16.cpp

tests/gtests/internals/CMakeFiles/test_internals.dir/test_bfloat16.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/test_internals.dir/test_bfloat16.cpp.i"
	cd /home/yunyi/Desktop/zyy-workspace/oneDNN/build/tests/gtests/internals && /usr/bin/aarch64-linux-gnu-g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/yunyi/Desktop/zyy-workspace/oneDNN/tests/gtests/internals/test_bfloat16.cpp > CMakeFiles/test_internals.dir/test_bfloat16.cpp.i

tests/gtests/internals/CMakeFiles/test_internals.dir/test_bfloat16.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/test_internals.dir/test_bfloat16.cpp.s"
	cd /home/yunyi/Desktop/zyy-workspace/oneDNN/build/tests/gtests/internals && /usr/bin/aarch64-linux-gnu-g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/yunyi/Desktop/zyy-workspace/oneDNN/tests/gtests/internals/test_bfloat16.cpp -o CMakeFiles/test_internals.dir/test_bfloat16.cpp.s

tests/gtests/internals/CMakeFiles/test_internals.dir/test_comparison_operators.cpp.o: tests/gtests/internals/CMakeFiles/test_internals.dir/flags.make
tests/gtests/internals/CMakeFiles/test_internals.dir/test_comparison_operators.cpp.o: /home/yunyi/Desktop/zyy-workspace/oneDNN/tests/gtests/internals/test_comparison_operators.cpp
tests/gtests/internals/CMakeFiles/test_internals.dir/test_comparison_operators.cpp.o: tests/gtests/internals/CMakeFiles/test_internals.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/yunyi/Desktop/zyy-workspace/oneDNN/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Building CXX object tests/gtests/internals/CMakeFiles/test_internals.dir/test_comparison_operators.cpp.o"
	cd /home/yunyi/Desktop/zyy-workspace/oneDNN/build/tests/gtests/internals && /usr/bin/aarch64-linux-gnu-g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT tests/gtests/internals/CMakeFiles/test_internals.dir/test_comparison_operators.cpp.o -MF CMakeFiles/test_internals.dir/test_comparison_operators.cpp.o.d -o CMakeFiles/test_internals.dir/test_comparison_operators.cpp.o -c /home/yunyi/Desktop/zyy-workspace/oneDNN/tests/gtests/internals/test_comparison_operators.cpp

tests/gtests/internals/CMakeFiles/test_internals.dir/test_comparison_operators.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/test_internals.dir/test_comparison_operators.cpp.i"
	cd /home/yunyi/Desktop/zyy-workspace/oneDNN/build/tests/gtests/internals && /usr/bin/aarch64-linux-gnu-g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/yunyi/Desktop/zyy-workspace/oneDNN/tests/gtests/internals/test_comparison_operators.cpp > CMakeFiles/test_internals.dir/test_comparison_operators.cpp.i

tests/gtests/internals/CMakeFiles/test_internals.dir/test_comparison_operators.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/test_internals.dir/test_comparison_operators.cpp.s"
	cd /home/yunyi/Desktop/zyy-workspace/oneDNN/build/tests/gtests/internals && /usr/bin/aarch64-linux-gnu-g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/yunyi/Desktop/zyy-workspace/oneDNN/tests/gtests/internals/test_comparison_operators.cpp -o CMakeFiles/test_internals.dir/test_comparison_operators.cpp.s

tests/gtests/internals/CMakeFiles/test_internals.dir/test_dnnl_threading.cpp.o: tests/gtests/internals/CMakeFiles/test_internals.dir/flags.make
tests/gtests/internals/CMakeFiles/test_internals.dir/test_dnnl_threading.cpp.o: /home/yunyi/Desktop/zyy-workspace/oneDNN/tests/gtests/internals/test_dnnl_threading.cpp
tests/gtests/internals/CMakeFiles/test_internals.dir/test_dnnl_threading.cpp.o: tests/gtests/internals/CMakeFiles/test_internals.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/yunyi/Desktop/zyy-workspace/oneDNN/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Building CXX object tests/gtests/internals/CMakeFiles/test_internals.dir/test_dnnl_threading.cpp.o"
	cd /home/yunyi/Desktop/zyy-workspace/oneDNN/build/tests/gtests/internals && /usr/bin/aarch64-linux-gnu-g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT tests/gtests/internals/CMakeFiles/test_internals.dir/test_dnnl_threading.cpp.o -MF CMakeFiles/test_internals.dir/test_dnnl_threading.cpp.o.d -o CMakeFiles/test_internals.dir/test_dnnl_threading.cpp.o -c /home/yunyi/Desktop/zyy-workspace/oneDNN/tests/gtests/internals/test_dnnl_threading.cpp

tests/gtests/internals/CMakeFiles/test_internals.dir/test_dnnl_threading.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/test_internals.dir/test_dnnl_threading.cpp.i"
	cd /home/yunyi/Desktop/zyy-workspace/oneDNN/build/tests/gtests/internals && /usr/bin/aarch64-linux-gnu-g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/yunyi/Desktop/zyy-workspace/oneDNN/tests/gtests/internals/test_dnnl_threading.cpp > CMakeFiles/test_internals.dir/test_dnnl_threading.cpp.i

tests/gtests/internals/CMakeFiles/test_internals.dir/test_dnnl_threading.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/test_internals.dir/test_dnnl_threading.cpp.s"
	cd /home/yunyi/Desktop/zyy-workspace/oneDNN/build/tests/gtests/internals && /usr/bin/aarch64-linux-gnu-g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/yunyi/Desktop/zyy-workspace/oneDNN/tests/gtests/internals/test_dnnl_threading.cpp -o CMakeFiles/test_internals.dir/test_dnnl_threading.cpp.s

tests/gtests/internals/CMakeFiles/test_internals.dir/__/main.cpp.o: tests/gtests/internals/CMakeFiles/test_internals.dir/flags.make
tests/gtests/internals/CMakeFiles/test_internals.dir/__/main.cpp.o: /home/yunyi/Desktop/zyy-workspace/oneDNN/tests/gtests/main.cpp
tests/gtests/internals/CMakeFiles/test_internals.dir/__/main.cpp.o: tests/gtests/internals/CMakeFiles/test_internals.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/yunyi/Desktop/zyy-workspace/oneDNN/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_5) "Building CXX object tests/gtests/internals/CMakeFiles/test_internals.dir/__/main.cpp.o"
	cd /home/yunyi/Desktop/zyy-workspace/oneDNN/build/tests/gtests/internals && /usr/bin/aarch64-linux-gnu-g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT tests/gtests/internals/CMakeFiles/test_internals.dir/__/main.cpp.o -MF CMakeFiles/test_internals.dir/__/main.cpp.o.d -o CMakeFiles/test_internals.dir/__/main.cpp.o -c /home/yunyi/Desktop/zyy-workspace/oneDNN/tests/gtests/main.cpp

tests/gtests/internals/CMakeFiles/test_internals.dir/__/main.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/test_internals.dir/__/main.cpp.i"
	cd /home/yunyi/Desktop/zyy-workspace/oneDNN/build/tests/gtests/internals && /usr/bin/aarch64-linux-gnu-g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/yunyi/Desktop/zyy-workspace/oneDNN/tests/gtests/main.cpp > CMakeFiles/test_internals.dir/__/main.cpp.i

tests/gtests/internals/CMakeFiles/test_internals.dir/__/main.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/test_internals.dir/__/main.cpp.s"
	cd /home/yunyi/Desktop/zyy-workspace/oneDNN/build/tests/gtests/internals && /usr/bin/aarch64-linux-gnu-g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/yunyi/Desktop/zyy-workspace/oneDNN/tests/gtests/main.cpp -o CMakeFiles/test_internals.dir/__/main.cpp.s

tests/gtests/internals/CMakeFiles/test_internals.dir/__/__/test_thread.cpp.o: tests/gtests/internals/CMakeFiles/test_internals.dir/flags.make
tests/gtests/internals/CMakeFiles/test_internals.dir/__/__/test_thread.cpp.o: /home/yunyi/Desktop/zyy-workspace/oneDNN/tests/test_thread.cpp
tests/gtests/internals/CMakeFiles/test_internals.dir/__/__/test_thread.cpp.o: tests/gtests/internals/CMakeFiles/test_internals.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/yunyi/Desktop/zyy-workspace/oneDNN/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_6) "Building CXX object tests/gtests/internals/CMakeFiles/test_internals.dir/__/__/test_thread.cpp.o"
	cd /home/yunyi/Desktop/zyy-workspace/oneDNN/build/tests/gtests/internals && /usr/bin/aarch64-linux-gnu-g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT tests/gtests/internals/CMakeFiles/test_internals.dir/__/__/test_thread.cpp.o -MF CMakeFiles/test_internals.dir/__/__/test_thread.cpp.o.d -o CMakeFiles/test_internals.dir/__/__/test_thread.cpp.o -c /home/yunyi/Desktop/zyy-workspace/oneDNN/tests/test_thread.cpp

tests/gtests/internals/CMakeFiles/test_internals.dir/__/__/test_thread.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/test_internals.dir/__/__/test_thread.cpp.i"
	cd /home/yunyi/Desktop/zyy-workspace/oneDNN/build/tests/gtests/internals && /usr/bin/aarch64-linux-gnu-g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/yunyi/Desktop/zyy-workspace/oneDNN/tests/test_thread.cpp > CMakeFiles/test_internals.dir/__/__/test_thread.cpp.i

tests/gtests/internals/CMakeFiles/test_internals.dir/__/__/test_thread.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/test_internals.dir/__/__/test_thread.cpp.s"
	cd /home/yunyi/Desktop/zyy-workspace/oneDNN/build/tests/gtests/internals && /usr/bin/aarch64-linux-gnu-g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/yunyi/Desktop/zyy-workspace/oneDNN/tests/test_thread.cpp -o CMakeFiles/test_internals.dir/__/__/test_thread.cpp.s

# Object files for target test_internals
test_internals_OBJECTS = \
"CMakeFiles/test_internals.dir/test_bcast_strategy.cpp.o" \
"CMakeFiles/test_internals.dir/test_bfloat16.cpp.o" \
"CMakeFiles/test_internals.dir/test_comparison_operators.cpp.o" \
"CMakeFiles/test_internals.dir/test_dnnl_threading.cpp.o" \
"CMakeFiles/test_internals.dir/__/main.cpp.o" \
"CMakeFiles/test_internals.dir/__/__/test_thread.cpp.o"

# External object files for target test_internals
test_internals_EXTERNAL_OBJECTS =

tests/gtests/internals/test_internals: tests/gtests/internals/CMakeFiles/test_internals.dir/test_bcast_strategy.cpp.o
tests/gtests/internals/test_internals: tests/gtests/internals/CMakeFiles/test_internals.dir/test_bfloat16.cpp.o
tests/gtests/internals/test_internals: tests/gtests/internals/CMakeFiles/test_internals.dir/test_comparison_operators.cpp.o
tests/gtests/internals/test_internals: tests/gtests/internals/CMakeFiles/test_internals.dir/test_dnnl_threading.cpp.o
tests/gtests/internals/test_internals: tests/gtests/internals/CMakeFiles/test_internals.dir/__/main.cpp.o
tests/gtests/internals/test_internals: tests/gtests/internals/CMakeFiles/test_internals.dir/__/__/test_thread.cpp.o
tests/gtests/internals/test_internals: tests/gtests/internals/CMakeFiles/test_internals.dir/build.make
tests/gtests/internals/test_internals: src/libdnnl.so.3.1
tests/gtests/internals/test_internals: tests/gtests/gtest/libdnnl_gtest.a
tests/gtests/internals/test_internals: tests/gtests/internals/CMakeFiles/test_internals.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/yunyi/Desktop/zyy-workspace/oneDNN/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_7) "Linking CXX executable test_internals"
	cd /home/yunyi/Desktop/zyy-workspace/oneDNN/build/tests/gtests/internals && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/test_internals.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
tests/gtests/internals/CMakeFiles/test_internals.dir/build: tests/gtests/internals/test_internals
.PHONY : tests/gtests/internals/CMakeFiles/test_internals.dir/build

tests/gtests/internals/CMakeFiles/test_internals.dir/clean:
	cd /home/yunyi/Desktop/zyy-workspace/oneDNN/build/tests/gtests/internals && $(CMAKE_COMMAND) -P CMakeFiles/test_internals.dir/cmake_clean.cmake
.PHONY : tests/gtests/internals/CMakeFiles/test_internals.dir/clean

tests/gtests/internals/CMakeFiles/test_internals.dir/depend:
	cd /home/yunyi/Desktop/zyy-workspace/oneDNN/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/yunyi/Desktop/zyy-workspace/oneDNN /home/yunyi/Desktop/zyy-workspace/oneDNN/tests/gtests/internals /home/yunyi/Desktop/zyy-workspace/oneDNN/build /home/yunyi/Desktop/zyy-workspace/oneDNN/build/tests/gtests/internals /home/yunyi/Desktop/zyy-workspace/oneDNN/build/tests/gtests/internals/CMakeFiles/test_internals.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : tests/gtests/internals/CMakeFiles/test_internals.dir/depend

