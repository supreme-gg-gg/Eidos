# Add all .cpp files in the src directory (excluding main.cpp)
file(GLOB SOURCES "*.cpp")
list(REMOVE_ITEM SOURCES "${CMAKE_SOURCE_DIR}/src/main.cpp")  # Exclude main.cpp from SOURCES

add_executable(main "main.cpp" ${SOURCES})

# Set the number of threads to use in Eigen (optional)
set(EIGEN_NUM_THREADS 8)  # Set to the number of threads you want
add_definitions(-DEIGEN_NUM_THREADS=${EIGEN_NUM_THREADS})

# Link Eigen3 to the main executable
target_link_libraries(main PRIVATE Eigen3::Eigen)

# Add option for compiling demo executables
option(BUILD_DEMO_EXECUTABLES "Build demo executables" OFF)

# If the option is ON, compile the demo executables
if(BUILD_DEMO_EXECUTABLES)
    message(STATUS "Building demo executables.")

    # Create three demo executables with sources from src, excluding main.cpp
    add_executable(run_mlp "demo/run_mlp.cpp" ${SOURCES})
    # add_executable(run_mnist "demo/run_mnist.cpp" ${SOURCES})
    add_executable(run_rnn "demo/run_rnn.cpp" ${SOURCES})

    # Link Eigen3 and OpenMP to the demo executables
    target_link_libraries(run_mlp PRIVATE Eigen3::Eigen)
    # target_link_libraries(run_mnist PRIVATE Eigen3::Eigen OpenMP::OpenMP_CXX)
    target_link_libraries(run_rnn PRIVATE Eigen3::Eigen)
else()
    message(STATUS "Not building demo executables.")
endif()

# Find OpenMP
find_package(OpenMP)

# Check if OpenMP is found and link it to the project
if(OpenMP_CXX_FOUND)
    message(STATUS "Found OpenMP, enabling multi-threading.")
    target_compile_definitions(main PRIVATE EIGEN_USE_OPENMP)  # Enable OpenMP in Eigen
    target_link_libraries(main PRIVATE OpenMP::OpenMP_CXX)    # Link OpenMP
    target_link_libraries(run_mlp PRIVATE OpenMP::OpenMP_CXX)    # Link OpenMP
    # target_link_libraries(run_mnist PRIVATE OpenMP::OpenMP_CXX)    # Link OpenMP
    target_link_libraries(run_rnn PRIVATE OpenMP::OpenMP_CXX)    # Link OpenMP
else()
    message(WARNING "OpenMP not found, running without multi-threading.")
endif()