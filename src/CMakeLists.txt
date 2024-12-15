# Add all .cpp files in the src directory
file(GLOB SOURCES "*.cpp")

add_executable(main ${SOURCES})

# Find OpenMP
find_package(OpenMP)

# Check if OpenMP is found and link it to the project
if(OpenMP_CXX_FOUND)
    message(STATUS "Found OpenMP, enabling multi-threading.")
    target_compile_definitions(main PRIVATE EIGEN_USE_OPENMP)  # Enable OpenMP in Eigen
    target_link_libraries(main PRIVATE OpenMP::OpenMP_CXX)    # Link OpenMP
else()
    message(WARNING "OpenMP not found, running without multi-threading.")
endif()

# Set the number of threads to use in Eigen (optional)
set(EIGEN_NUM_THREADS 8)  # Set to the number of threads you want
add_definitions(-DEIGEN_NUM_THREADS=${EIGEN_NUM_THREADS})

# Link Eigen3 to the main executable
target_link_libraries(main PRIVATE Eigen3::Eigen)
