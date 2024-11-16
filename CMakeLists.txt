# Specify the minimum CMake version
cmake_minimum_required(VERSION 3.10)

# Set the project name and language
project(ComputerVisionCpp)

# Enable C++14 or later
set(CMAKE_CXX_STANDARD 14)

find_package(Eigen3 REQUIRED)

# !!! PLEASE MODIFY IF YOU USE A CUSTOM PATH
if (NOT Eigen3_FOUND)
    set(Eigen3_DIR "~/eigen3/cmake" CACHE PATH "Path to Eigen3 CMake configuration")
    find_package(Eigen3 REQUIRED)
endif()

# If still not found, allow a custom path via environment variable or cache
if (NOT Eigen3_FOUND)
    message(FATAL_ERROR "Eigen3 not found. Please specify Eigen3_DIR via CMake or set the correct path.")
endif()

# Add the src directory to the build process
add_subdirectory(src)
