# Eidos: Deep Learning Library from Scratch in C++

## Description

This is a project for educational purpose to implement MLP, CNN, RNN and variants and helper functions (e.g. data loader) from scratch using only Eigen for parallelized linear algebra in pure C++. The goal is to understand the underlying concepts of these models and how they work. The project is inspired by the libtorch, PyTorch's C++ frontend. We aim to provide a similar API interface to PyTorch and Keras for ease of use.

> The models are not optimized for performance like PyTorch. Thus, it will take (considerably) longer to train, especially for CNN. However, we gurantee that performance-wise, on tasks of reasonable complexity, it will be simliar to PyTorch.

## Installation

### Build From Source (Recommended)

The following method will compile and install the library to your local path and allows you to use the library.

1. Clone the repo from [our GitHub](https://github.com/supreme-gg-gg/deep-learning-cpp)

2. Create a build directory: `mkdir build`

3. Navigate to `build/` and run: `cmake .. && make`

4. Install the library with `make install`

### Pre-built Binaries

This is only provided for certain platforms...

### Using the installed library

1. In your source code, include the main header `#include <Eidos/Eidos.h>`

2. You can compile it by manually linking it with the compilers' `-I/path/to/installed/Eidos -I/path/to/Eigen -L/path/to/installed/lib -lEidos`

3. Or (recommended) use cmake's `find_package(Eidos REQUIERD)`. A sample is provided in `demo/`

### Running Tests

This project uses Google Test and ctest for unit testing.

To run the unit tests, follow these steps:

1. Follow instructions under [build from source](#build-from-source-recommended)

2. Run cmake command with the build test flag: `cmake .. -DBUILD_TESTS=ON`

3. In the `build/tests` directory, run `make` to compile the tests

4. Run tests with either `./run_tests` or `ctest`

> Note that a few loss function UTs are failing for some reason under investigation...

## API Interface

Usage of our library is extremely similar to PyTorch and Keras to make it easy for python ML practioneres. The detailed documentation is provided in `docs/`. Sample usage are provided in `demo/`

## About our name

Eidos...

## Contributing

Please contact the owner for contribution. 

## Credits

Jet Chiang, James Zheng
