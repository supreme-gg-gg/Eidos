cmake_minimum_required(VERSION 3.10)

# Enable C++17 or later
set(CMAKE_CXX_STANDARD 17)

project(Eidos VERSION 1.0.0 LANGUAGES CXX)

# Enable debug mode (if specified)
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

# Find Eigen3
find_package(Eigen3 3.3 REQUIRED)

# Find OpenMP (if available)
find_package(OpenMP)

# Set up the source files for the core library
file(GLOB SOURCES "src/*.cpp")
list(REMOVE_ITEM SOURCES "${CMAKE_SOURCE_DIR}/src/main.cpp")  # Exclude main.cpp if it exists

# Create the core library
add_library(Eidos ${SOURCES})

# Set Eigen thread count (optional)
set(EIGEN_NUM_THREADS 8)  # Adjust based on your needs
add_definitions(-DEIGEN_NUM_THREADS=${EIGEN_NUM_THREADS})

# Link Eigen and OpenMP (if available)
target_link_libraries(Eidos PRIVATE Eigen3::Eigen)
if(OpenMP_CXX_FOUND)
    target_compile_definitions(Eidos PRIVATE EIGEN_USE_OPENMP)
    target_link_libraries(Eidos PRIVATE OpenMP::OpenMP_CXX)
else()
    message(WARNING "OpenMP not found, running without multi-threading.")
endif()

# Set include directories for the library
target_include_directories(Eidos
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

# Install the core library and headers
install(
    TARGETS Eidos
    EXPORT EidosTargets
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR} # Shared library
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR} # Static library
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR} # Executables
    INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

install(
    DIRECTORY include/Eidos
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

# Generate and install CMake config files for find_package
include(CMakePackageConfigHelpers)
write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/EidosConfigVersion.cmake"
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY AnyNewerVersion
)

# Configure EidosConfig.cmake file
configure_package_config_file(
    "${CMAKE_CURRENT_LIST_DIR}/cmake/EidosConfig.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}/EidosConfig.cmake"
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/Eidos
)

install(
    FILES "${CMAKE_CURRENT_BINARY_DIR}/EidosConfig.cmake"
          "${CMAKE_CURRENT_BINARY_DIR}/EidosConfigVersion.cmake"
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/Eidos
)

# Install the EidosTargets for downstream projects
install(
    EXPORT EidosTargets
    FILE EidosTargets.cmake
    NAMESPACE Eidos::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/Eidos
)

# Export targets to use in the build tree
export(
    EXPORT EidosTargets
    FILE "${CMAKE_CURRENT_BINARY_DIR}/EidosTargets.cmake"
    NAMESPACE Eidos::
)