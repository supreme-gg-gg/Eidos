# Specify the minimum CMake version
cmake_minimum_required(VERSION 3.10)

# Set the project name and language
project(ComputerVisionCpp)

# Enable C++14 or later
set(CMAKE_CXX_STANDARD 14)

# Add the src directory to the build process
add_subdirectory(src)
