#include "../include/conv_layer.h"
#include <cmath>

ConvLayer::ConvLayer(int in_channels, int out_channels, int kernel_size, int stride, int padding)
    : in_channels(in_channels), out_channels(out_channels), kernel_size(kernel_size), stride(stride), padding(padding) {
    // Initialize kernels and biases
    for (int i = 0; i < out_channels; ++i) {
        kernels.push_back(Eigen::MatrixXf::Random(in_channels, kernel_size * kernel_size) * sqrt(2.0f / in_channels));
        grad_kernels.push_back(Eigen::MatrixXf::Zero(in_channels, kernel_size * kernel_size));
    }
    bias = Eigen::VectorXf::Zero(out_channels);
    grad_bias = Eigen::VectorXf::Zero(out_channels);
}

std::vector<Eigen::MatrixXf> ConvLayer::forward(const Eigen::MatrixXf& input) {
    // Get input dimensions
    int height = input.rows();
    int width = input.cols();

    // Calculate output dimensions
    int out_height = (height + 2 * padding - kernel_size) / stride + 1;
    int out_width = (width + 2 * padding - kernel_size) / stride + 1;

    // Initialize output
    Eigen::MatrixXf output = Eigen::MatrixXf::Zero(out_height * out_width, out_channels);

    // Initialize caches
    input_cache.clear();
    output_cache.clear();

    // Pad input
    Eigen::MatrixXf padded_input = Eigen::MatrixXf::Zero(height + 2 * padding, width + 2 * padding);
    padded_input.block(padding, padding, height, width) = input;

    // Slide kernel over input
    for (int i = 0; i < out_height; ++i) {
        for (int j = 0; j < out_width; ++j) {
            // Extract receptive field
            Eigen::MatrixXf receptive_field = padded_input.block(i * stride, j * stride, kernel_size, kernel_size);

            // Flatten receptive field
            Eigen::VectorXf flattened_receptive_field = receptive_field.transpose().reshape(kernel_size * kernel_size, 1);

            // Perform convolution
            for (int k = 0; k < out_channels; ++k) {
                output(i * out_width + j, k) = (kernels[k] * flattened_receptive_field)(0) + bias(k);
            }

            // Cache input and output
            input_cache.push_back(flattened_receptive_field);
            output_cache.push_back(output.block(i * out_width + j, 0, 1, out_channels));
        }
    }

    return output;
}