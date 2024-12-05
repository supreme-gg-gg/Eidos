# Find Eigen3 package (automatically looks in default locations like /usr/include/eigen3)
find_package(Eigen3 REQUIRED)

# Add all .cpp files in the src directory
file(GLOB SOURCES "*.cpp")

# add_executable(main ${SOURCES})
add_executable(main 
main.cpp
cross_entropy_loss.cpp
mse_loss.cpp
dense_layer.cpp
optimizer.cpp
)

# Link Eigen3 to the main executable
target_link_libraries(main PRIVATE Eigen3::Eigen)

# Optionally, link other libraries (e.g., pthreads if needed)
# target_link_libraries(main PRIVATE pthread)
