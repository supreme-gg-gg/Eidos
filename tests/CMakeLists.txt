cmake_minimum_required(VERSION 3.10)

enable_testing()

set(CMAKE_CXX_STANDARD 17)

project(Eidos_Tests LANGUAGES CXX)

# Google Test setup (if using FetchContent)
include(FetchContent)
FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG        release-1.11.0  # Stable release version
)
FetchContent_MakeAvailable(googletest)

# Include the Google Test framework
include_directories(${gtest_SOURCE_DIR}/include ${gtest_SOURCE_DIR})

# Test files
# When you have more dependencies in src
# you can just search them using GLOB too
file(GLOB TEST_FILES
    "test_*.cpp"
    "../src/*.cpp"
)

# Add the test executable and link it with Google Test, and your project library (computer_vision_lib)
add_executable(run_tests ${TEST_FILES})

find_package(Eigen3 3.3 REQUIRED NO_MODULE)

# Link Google Test and your project library (this is where the source code is linked)
target_link_libraries(run_tests gtest_main Eigen3::Eigen)

# Discover tests automatically (this is helpful for CTest)
include(GoogleTest)
gtest_discover_tests(run_tests)