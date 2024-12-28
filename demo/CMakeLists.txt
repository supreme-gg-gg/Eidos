cmake_minimum_required(VERSION 3.10)

# Project name and language
project(MyApp VERSION 1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)

# Find the installed Eidos library using find_package
# This assumes that Eidos was installed in a directory where CMake can find it.
# If not, the user can set CMAKE_PREFIX_PATH to the directory where the library is installed.
find_package(Eidos REQUIRED)
find_package(Eigen3 3.3 REQUIRED)

# Add the user's source files (main.cpp in this case)
add_executable(my_app main.cpp)

# Link the Eidos library to the executable
target_link_libraries(my_app PRIVATE Eidos)
target_link_libraries(my_app PRIVATE Eigen3::Eigen)

message(STATUS "Eidos include directories: ${Eidos_INCLUDE_DIRS}")
message(STATUS "Eidos library directories: ${Eidos_LIBRARY_DIRS}")

# Specify include directories if necessary
target_include_directories(my_app PRIVATE /usr/local/include)
target_link_directories(my_app PRIVATE /usr/local/lib)
