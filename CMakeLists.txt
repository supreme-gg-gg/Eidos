cmake_minimum_required(VERSION 3.10)

# Enable C++17 or later
set(CMAKE_CXX_STANDARD 17)

project(Eidos VERSION 1.0.0 LANGUAGES CXX)

# Add source files for the library
# add_subdirectory(src)

# Add the tests directory to the build process (but only if tests are enabled)
option(BUILD_TESTS "Build the test suite" OFF)
if(BUILD_TESTS)
    add_subdirectory(tests)
endif()

# Enable testing
enable_testing()

# Provide an option for shared/static library
option(BUILD_SHARED_LIBS "Build shared libraries instead of static libraries" ON)

# Check for Eigen3
find_package(Eigen3 3.3 REQUIRED)

# Find OpenMP
find_package(OpenMP)

option(DEBUG_MODE "Enable debug mode" ON)

if(DEBUG_MODE)
    message(STATUS "Debug mode is enabled.")
    set(CMAKE_BUILD_TYPE Debug)
    add_definitions(-DDEBUG_MODE)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=address -g")
    set(CMAKE_LINKER_FLAGS "${CMAKE_LINKER_FLAGS} -fsanitize=address")
else()
    message(STATUS "Debug mode is disabled.")
endif()

# Install setup
include(GNUInstallDirs)

# Only include source files for the core library (no main executable)
file(GLOB SOURCES "src/*.cpp")
list(REMOVE_ITEM SOURCES "${CMAKE_SOURCE_DIR}/src/main.cpp")  # Exclude main.cpp

# Add the core library target
add_library(Eidos ${SOURCES})

# Set the number of threads to use in Eigen (optional)
set(EIGEN_NUM_THREADS 8)  # Set to the number of threads you want
add_definitions(-DEIGEN_NUM_THREADS=${EIGEN_NUM_THREADS})

# Link Eigen3 and OpenMP to the core library
target_link_libraries(Eidos PRIVATE Eigen3::Eigen)
if(OpenMP_CXX_FOUND)
    target_compile_definitions(Eidos PRIVATE EIGEN_USE_OPENMP)  # Enable OpenMP in Eigen
    target_link_libraries(Eidos PRIVATE OpenMP::OpenMP_CXX)      # Link OpenMP
else()
    message(WARNING "OpenMP not found, running without multi-threading.")
endif()

# Install the library
install(
    TARGETS Eidos
    EXPORT EidosTargets
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR} # Shared library
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR} # Static library
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR} # Executables
    INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

# Install header files
install(
    DIRECTORY include/Eidos  # Assumes headers are in the root/include directory
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

# Export the library target for downstream use
install(
    EXPORT EidosTargets
    FILE EidosTargets.cmake
    NAMESPACE Eidos::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/Eidos
)

# Generate a config file for find_package
include(CMakePackageConfigHelpers)
write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/EidosConfigVersion.cmake"
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY AnyNewerVersion
)

configure_package_config_file(
    "${CMAKE_CURRENT_LIST_DIR}/cmake/EidosConfig.cmake.in" # Template file
    "${CMAKE_CURRENT_BINARY_DIR}/EidosConfig.cmake"
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/Eidos
)

install(
    FILES
        "${CMAKE_CURRENT_BINARY_DIR}/EidosConfig.cmake"
        "${CMAKE_CURRENT_BINARY_DIR}/EidosConfigVersion.cmake"
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/Eidos
)

# Export build tree for immediate use after building
export(
    EXPORT EidosTargets
    FILE "${CMAKE_CURRENT_BINARY_DIR}/EidosTargets.cmake"
    NAMESPACE Eidos::
)