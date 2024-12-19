#include "../include/layers/flatten_layer.h"
#include "../include/tensor.hpp"
#include <stdexcept>
#include <Eigen/Dense>

// Forward pass: Flattens the input tensor into a 1D vector within a 3D structure
Tensor FlattenLayer::forward(const Tensor& input) {
   // Get the shape of the input tensor (assuming it's 3D)
    // Assuming shape is a tuple (c, h, w)
    auto shape = input.shape();  // (c, h, w)

    // Accessing the individual elements of the tuple using std::get
    int c = std::get<0>(shape);  // channels
    int h = std::get<1>(shape);  // height
    int w = std::get<2>(shape);  // width

    // Calculate the flattened size
    int flattened_size = c * h * w;

    this->input_shape = {c, h, w};
    this->output_shape = {1, 1, flattened_size};

    // Create an Eigen Matrix to store the flattened data
    Eigen::MatrixXf flattened(1, flattened_size);

    // Manually flattening with a loop (iterate through each matrix in the vector)
    size_t idx = 0;
    for (size_t i = 0; i < c; ++i) {
        Eigen::MatrixXf matrix = input[i];  // Access each matrix (channel)
        Eigen::Map<Eigen::MatrixXf> flattened_map(flattened.data() + idx, 1, matrix.size());
        flattened_map = matrix.transpose().reshaped(1, matrix.size()); // Flatten each matrix (channel)
        idx += matrix.size();
    }

    // Return the flattened tensor as a 3D tensor (1, 1, flattened_size)
    return Tensor(flattened);
}

// Backward pass: Compute the gradient of the loss with respect to the input tensor
Tensor FlattenLayer::backward(const Tensor& grad_output) {
    // Get the gradient of the loss with respect to the output (flattened size)
    Eigen::MatrixXf grad_data = grad_output.getSingleMatrix();

    // Reshape the gradient to match the input shape of the flatten layer
    int c = input_shape[0];  // Number of channels
    int h = input_shape[1];  // Height of the input
    int w = input_shape[2];  // Width of the input

    Tensor grad_input(c, h, w);

    size_t idx = 0;
    for (int i = 0; i < c; ++i) {
        Eigen::MatrixXf matrix = grad_output.getSingleMatrix().block(i * h, 0, h, w);  // Extract the gradient channel
        grad_input[i] = matrix.transpose().reshaped(h, w);  // Reshape the gradient to match the input shape
    }

    // Return the reshaped gradient as a tensor
    return grad_input;
}