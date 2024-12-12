cmake_minimum_required(VERSION 3.10)

# Enable C++17 or later
set(CMAKE_CXX_STANDARD 17)

# Find Eigen3
find_package(Eigen3 REQUIRED)

# Google Test setup (if using FetchContent)
include(FetchContent)
FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG        release-1.11.0  # Stable release version
)
FetchContent_MakeAvailable(googletest)

# Add the src directory to the build process
add_subdirectory(src)

# Add the tests directory to the build process
add_subdirectory(tests)

# Enable testing
enable_testing()