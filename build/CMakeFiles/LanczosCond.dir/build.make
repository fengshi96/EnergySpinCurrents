# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.30

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
CMAKE_COMMAND = /Applications/CMake.app/Contents/bin/cmake

# The command to remove a file.
RM = /Applications/CMake.app/Contents/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /Users/shifeng/Projects/5.SpinEnergyCurrents/LanczosCond

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /Users/shifeng/Projects/5.SpinEnergyCurrents/LanczosCond/build

# Include any dependencies generated for this target.
include CMakeFiles/LanczosCond.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/LanczosCond.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/LanczosCond.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/LanczosCond.dir/flags.make

CMakeFiles/LanczosCond.dir/main.cpp.o: CMakeFiles/LanczosCond.dir/flags.make
CMakeFiles/LanczosCond.dir/main.cpp.o: /Users/shifeng/Projects/5.SpinEnergyCurrents/LanczosCond/main.cpp
CMakeFiles/LanczosCond.dir/main.cpp.o: CMakeFiles/LanczosCond.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/Users/shifeng/Projects/5.SpinEnergyCurrents/LanczosCond/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/LanczosCond.dir/main.cpp.o"
	/usr/bin/clang++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/LanczosCond.dir/main.cpp.o -MF CMakeFiles/LanczosCond.dir/main.cpp.o.d -o CMakeFiles/LanczosCond.dir/main.cpp.o -c /Users/shifeng/Projects/5.SpinEnergyCurrents/LanczosCond/main.cpp

CMakeFiles/LanczosCond.dir/main.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/LanczosCond.dir/main.cpp.i"
	/usr/bin/clang++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Users/shifeng/Projects/5.SpinEnergyCurrents/LanczosCond/main.cpp > CMakeFiles/LanczosCond.dir/main.cpp.i

CMakeFiles/LanczosCond.dir/main.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/LanczosCond.dir/main.cpp.s"
	/usr/bin/clang++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Users/shifeng/Projects/5.SpinEnergyCurrents/LanczosCond/main.cpp -o CMakeFiles/LanczosCond.dir/main.cpp.s

# Object files for target LanczosCond
LanczosCond_OBJECTS = \
"CMakeFiles/LanczosCond.dir/main.cpp.o"

# External object files for target LanczosCond
LanczosCond_EXTERNAL_OBJECTS =

LanczosCond: CMakeFiles/LanczosCond.dir/main.cpp.o
LanczosCond: CMakeFiles/LanczosCond.dir/build.make
LanczosCond: CMakeFiles/LanczosCond.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --bold --progress-dir=/Users/shifeng/Projects/5.SpinEnergyCurrents/LanczosCond/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable LanczosCond"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/LanczosCond.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/LanczosCond.dir/build: LanczosCond
.PHONY : CMakeFiles/LanczosCond.dir/build

CMakeFiles/LanczosCond.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/LanczosCond.dir/cmake_clean.cmake
.PHONY : CMakeFiles/LanczosCond.dir/clean

CMakeFiles/LanczosCond.dir/depend:
	cd /Users/shifeng/Projects/5.SpinEnergyCurrents/LanczosCond/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /Users/shifeng/Projects/5.SpinEnergyCurrents/LanczosCond /Users/shifeng/Projects/5.SpinEnergyCurrents/LanczosCond /Users/shifeng/Projects/5.SpinEnergyCurrents/LanczosCond/build /Users/shifeng/Projects/5.SpinEnergyCurrents/LanczosCond/build /Users/shifeng/Projects/5.SpinEnergyCurrents/LanczosCond/build/CMakeFiles/LanczosCond.dir/DependInfo.cmake "--color=$(COLOR)"
.PHONY : CMakeFiles/LanczosCond.dir/depend

