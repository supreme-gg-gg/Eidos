cmake_minimum_required(VERSION 3.10)

project(MyCppProject)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)

# Add include directory
include_directories(${PROJECT_SOURCE_DIR}/include)

# Add the source directory
add_subdirectory(src)

# Optionally, add tests
enable_testing()
add_subdirectory(tests)
