# Eidos: Deep Learning Library from Scratch in C++

## Description

Eidos is a project for educational purpose to build an entire deep learning library from scratch in C++. It implemented MLP, CNN, RNN and helper utilities (e.g. data loader) using only Eigen for parallelized linear algebra.

The goal is to understand the underlying concepts of these models and how they work. The project is inspired by the libtorch, PyTorch's C++ frontend. We aim to provide a similar API interface to PyTorch and Keras for ease of use.

> The models are not optimized for performance like PyTorch. Thus, it will take (considerably) longer to train.

## Installation

The library is built using CMake and requires Eigen3. Eigen is a C++ template library for parallelized linear algebra: matrices, vectors, numerical solvers, and related algorithms.

You can install Eigen3 using your package manager or download it from the [official website](https://eigen.tuxfamily.org/index.php?title=Main_Page). You can install cmake using your package manager or download it from the [official website](https://cmake.org/).

### Build From Source (Recommended)

The following method will compile and install the library to your local path and allows you to use the library.

First, clone the repository from GitHub. Then follow these steps:

```sh
mkdir build && cd build
cmake .. && make
sudo make install
```

You can enable debug mode by adding `-DDEBUG_MODE=ON` to the cmake command. This will setup utilities such as address sanitizer.

### Pre-built Binaries

Pre-built binaries are supported for MacOS, Linux and Windows. You can download the binaries from the [releases page]()

After downloading the tarball, extract it to your desired location:

```sh
# For Linux and MacOS
tar -xzvf Eidos.tar.gz -C /desired/location/

# For Windows
unzip Eidos.zip -d C:\desired\location\
```

Add the library to your path:

```sh
# For Linux and MacOS
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/desired/location/lib

# For Windows
setx PATH "%PATH%;C:\desired\location\lib"
```

### Using the installed library

In your source code, include the main header `#include <Eidos/Eidos.h>`. Compile your source file by either:

1. Manual linking with compiler such as `g++ main.cpp -I/path/to/installed/Eidos -I/path/to/Eigen -L/path/to/installed/lib -lEidos`

2. Recommended: cmake's `find_package(Eidos REQUIERD)`. **A sample is provided in `demo/CMakeLists.txt`**

### Running Tests

This project uses Google Test for unit testing. The unit tests contained in `tests/` are basic and more like sanity checks. You do not need to install the library to run UTs. To run the tests, follow these steps:

```sh
cd tests && mkdir build && cd build
cmake .. && make
ctest # or ./run_tests
```

## API Interface

Usage of our library is extremely similar to PyTorch and Keras to make it easy for python ML practioneres. A very simple example for a MLP is shown below:

```cpp
#include <Eidos/Eidos.h>

int main() {

    // Create a label map for the data loader
    std::map<std::string, int> label_map;
    for (int i = 0; i < 10; ++i) {
        label_map[std::to_string(i)] = i;
    }

    // Load data and split into training and testing
    auto data = NumericDataLoader("mnist_train.csv", "label", label_map).train_test_split(0.8, 32);

    // Create model and add fully connected layers
    Model model;
    model.Add(new DenseLayer(784, 128));
    model.Add(new ReLU());
    model.Add(new DenseLayer(128, 10));

    // Create optimizer and loss function
    Adam optimizer(0.001);
    CrossEntropyLoss loss_fn;

    // Train and test the model using convenience API
    model.Train(data.training.inputs, data.training.targets, 5, &optimizer, &loss_fn);
    model.Test(data.testing.inputs, data.testing.targets, &loss_fn);

    // Save the model
    model.Serialize("myModel.bin");
}
```

You should consult the full documentation in `docs/` and sample code in `demo/` for more powerful features such as CNN, RNN, regularisation, custom training loops, data preprocessing, etc.

## About our name

> If understanding and true opinion are distinct, then these ‘by themselves’ things definitely exist – these Forms (Eidos), the objects not of our sense perception, but of understanding only. (Plato, Timaeus, 51d)

**"Eidos", a Greek word meaning "form" or "essence"**, is derived from Plato's Theory of the Forms, representing our pursuit of understanding the essential principles underlying machine learning. Just as forms in Plato’s philosophy are timeless, absolute truths behind the imperfect representations of the physical world, deep learning methods allow us to discern patterns and insights from complex data, transcending the mutable and flawed representations we observe. Our library, crafted from scratch as an educational journey, embodies this philosophical aspiration: not just to create, but to learn and uncover the foundational truths of AI.

## Contributing

We welcome contributions to the project. Please refer to the [CONTRIBUTING.md](./CONTRIBUTING.md) for more information.

## Credits

[Jet Chiang](https://www.linkedin.com/in/jet-chiang/), [James Zheng](https://www.linkedin.com/in/james-zheng-zi/)

Feel free to contact us at our LinkedIn profiles for any questions or feedback.
