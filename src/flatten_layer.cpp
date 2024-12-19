#include "../include/layers/flatten_layer.h"
#include "../tensor.hpp"
#include <stdexcept>
#include <Eigen/Dense>

FlattenLayer::FlattenLayer(const std::vector<int>& input_shape, const std::vector<int>& output_shape = {})
    : input_shape(input_shape) {
    if (input_shape.size() != 3) {
        throw std::invalid_argument("Input shape must be 3D (Channels, Height, Width).");
    }
    if (output_shape.empty()) {
        // Compute output shape: {Batch_Size, Flattened_Size}
        // default to flatten all except first dimension
        int flattened_size = input_shape[1] * input_shape[2]; 
        this->output_shape = {input_shape[0], flattened_size};
    } else if (output_shape.size() == 2) {
        this->output_shape = output_shape;
    } else {
        throw std::invalid_argument("Output shape must be 2D (Channels, Flattened_Size).");
    }
}

Tensor FlattenLayer::forward(const Tensor& input) {
    // Validate the input tensor shape
    if (input.depth() != input_shape[0]) {
        throw std::invalid_argument("Input tensor depth does not match expected number of channels.");
    }

    int batch_size = this->output_shape[0];             // Number of rows in each matrix
    int flattened_size = this->output_shape[1];  // Flattened size (b * c)

    Tensor output(1, batch_size, flattened_size);   // Create output tensor with appropriate shape

    // Flatten each matrix in the input tensor
    for (int i = 0; i < batch_size; ++i) {
        Eigen::VectorXf flattened(flattened_size);
        int offset = 0;

        for (size_t channel = 0; channel < input.depth(); ++channel) {
            flattened.segment(offset, input[channel].cols()) = input[channel].row(i);
            offset += input[channel].cols();
        }
        output[0].row(i) = flattened.transpose();
    }

    return output;
}

Tensor FlattenLayer::backward(const Tensor& grad) {
    int batch_size = grad[0].rows();             // Batch size from gradient tensor
    int flattened_size = grad[0].cols();         // Flattened size from gradient tensor

    // Validate gradient tensor shape
    if (flattened_size != output_shape[1]) {
        throw std::invalid_argument("Gradient size does not match expected flattened size.");
    }

    // Create reshaped gradient tensor with depth and each matrix having rows = batch_size and cols = input_shape[1] * input_shape[2]
    Tensor reshaped_grad(input_shape[0], input_shape[1], input_shape[2]);

    // Unpack the gradient tensor into its original shape
    for (int i = 0; i < batch_size; ++i) {
        int offset = 0;

        for (int channel = 0; channel < input_shape[0]; ++channel) {
            reshaped_grad[channel].row(i) = grad[0].segment(offset, input_shape[1] * input_shape[2]);
            offset += input_shape[1] * input_shape[2];
        }
    }

    return reshaped_grad;
}