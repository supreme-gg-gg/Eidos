enable_testing()

# Include the Google Test framework
include_directories(${gtest_SOURCE_DIR}/include ${gtest_SOURCE_DIR})

# Test files
# When you have more dependencies in src
# you can just search them using GLOB too
file(GLOB TEST_FILES
    "test_*.cpp"
    ../src/loss_fns.cpp
    ../src/dense_layer.cpp
    ../src/activation_fns.cpp
    ../src/optimizer.cpp
    ../src/conv_layer.cpp
)

# Add the test executable and link it with Google Test, and your project library (computer_vision_lib)
add_executable(run_tests ${TEST_FILES})

# Link Google Test and your project library (this is where the source code is linked)
target_link_libraries(run_tests gtest_main Eigen3::Eigen)

# Discover tests automatically (this is helpful for CTest)
include(GoogleTest)
gtest_discover_tests(run_tests)