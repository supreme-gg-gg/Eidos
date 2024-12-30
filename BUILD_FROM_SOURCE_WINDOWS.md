# Building the Library from Source on Windows with MSVC

This guide explains how to build `Eidos` from source on Windows using Microsoft Visual Studio (MSVC) and CMake.

## Prerequisites

Please refer to [README.md](README.md) for dependencies.
Also make sure you have Microsoft Visual Studio installed.

## Build Steps
1. Clone the library's repository to your local machine using Git:
    ```
    git clone https://github.com/supreme-gg-gg/eidos
    ```
2. Modify the `CMakeLists.txt` file as follows:
    - Add this line near the top:
      ```
      include(GNUInstallDirs)
      ```
    - Make sure debug mode is DISABLED (the linker will produce errors otherwise)
      ```
      option(DEBUG_MODE "Enable debug mode" OFF)
      ```
    Your edited `CMakeLists.txt` should look something like this:
    ```
    cmake_minimum_required(VERSION 3.10)
    
    # Enable C++17 or later
    set(CMAKE_CXX_STANDARD 17)
    
    project(Eidos VERSION 1.0.0 LANGUAGES CXX)
    
    include(GNUInstallDirs)
    
    # Enable debug mode (if specified)
    option(DEBUG_MODE "Enable debug mode" OFF)
    
    if(DEBUG_MODE)
        message(STATUS "Debug mode is enabled.")
        set(CMAKE_BUILD_TYPE Debug)
        add_definitions(-DDEBUG_MODE)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=address -g")
        set(CMAKE_LINKER_FLAGS "${CMAKE_LINKER_FLAGS} -fsanitize=address")
    else()
        message(STATUS "Debug mode is disabled.")
    endif()
    
    # ... rest of the file
    ```
3. Navigate into the repository's root directory and create a `build` subdirectory:
    ```
    cd eidos
    mkdir build
    cd build
    ```
    This `build` directory will hold all the generated files and build artifacts.
4. Run the following command inside the `build` directory to generate the Visual Studio solution and project files:
    ```
    cmake ..
    ```
5. Once the project files are generated, open the `eidos.sln` file in Visual Studio:
    ```
    .\eidos.sln
    ```
6. Select the desired build configuration (Release/Debug) and build the solution.
7. After building, the static library (`.lib` file) will be available in the `build\release` or `build\debug` directory. You can now link your projects to this `.lib` file.

Refer to the [README.md](README.md) file for details on using the library.