cmake_minimum_required(VERSION 3.10)
project(UnitTests)

# Automatically find all .cpp files in the test directory
file(GLOB TEST_SOURCES "*.cpp")

# Create the test executable
add_executable(run_tests
    main.cpp
    ../src/cross_entropy_loss.cpp
    ../src/mse_loss.cpp
)

# Optionally link any libraries needed for tests
target_link_libraries(run_tests PRIVATE Eigen3::Eigen)