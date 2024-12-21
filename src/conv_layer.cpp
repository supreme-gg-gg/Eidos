#include "../include/layers/conv_layer.h"
#include "../include/tensor.hpp"
#include <random>
#include <Eigen/Dense>

Conv2D::Conv2D(int input_channels, int output_channels, 
                                       int kernel_size, int stride, int padding)
    : kernel_size_(kernel_size), stride_(stride), padding_(padding) {
    // Store the input shape
    this->input_shape = {input_channels, -1, -1}; // Heights and widths are unknown at initialization.

    // Initialize weights, biases, and their gradients
    weights.resize(output_channels);
    biases.resize(output_channels);
    grad_weights.resize(output_channels);
    grad_biases.resize(output_channels);

    // Weight initialization using Xavier/Glorot Normal initialization
    for (int oc = 0; oc < output_channels; ++oc) {
        // Initialize the weights for each filter using Random method
        weights[oc] = Eigen::MatrixXf::Random(input_channels, kernel_size * kernel_size) 
            * std::sqrt(2.0f / (input_channels * kernel_size * kernel_size));

        // Initialize biases
        biases[oc] = Eigen::VectorXf::Zero(1); // Single bias per output channel
    }

    // Output shape calculation deferred to first forward pass based on input dimensions
}

std::vector<int> Conv2D::calculateOutputShape() const {
    // Ensure input_shape is valid
    if (this->input_shape.size() != 3) {
        throw std::invalid_argument("Input shape must have 3 dimensions: [channels, height, width].");
    }

    int C_in = this->input_shape[0];  // Number of input channels
    int H_in = this->input_shape[1];  // Height of the input
    int W_in = this->input_shape[2];  // Width of the input

    // Number of filters determines output channels
    int C_out = weights.size();  

    // Calculate output height and width
    int H_out = (H_in + 2 * padding_ - kernel_size_) / stride_ + 1;
    int W_out = (W_in + 2 * padding_ - kernel_size_) / stride_ + 1;

    // Return the output shape
    return {C_out, H_out, W_out};
}

Tensor Conv2D::applyPadding(const Tensor& input) {
    // Extract input dimensions
    int C_in = input_shape[0];  // Number of input channels
    int H_in = input_shape[1];  // Input height
    int W_in = input_shape[2];  // Input width

    // Calculate the padded height and width
    int H_padded = H_in + 2 * padding_;
    int W_padded = W_in + 2 * padding_;

    // Create a new tensor with the padded dimensions
    Tensor padded_input(C_in, H_padded, W_padded);

    // Copy the input data to the center of the padded tensor
    for (int c = 0; c < C_in; ++c) {
        padded_input[c].block(padding_, padding_, H_in, W_in) = input[c];
    }

    return padded_input;
}

Tensor Conv2D::forward(const Tensor& input) {
    // Cache the input for use in the backward pass
    this->cache_input = input;

    // Update the input shape based on the input dimensions
    this->input_shape[1] = input[0].rows();  // Height
    this->input_shape[2] = input[0].cols();  // Width

    // Calculate the output shape based on the input dimensions
    this->output_shape = calculateOutputShape();

    // Apply padding to the input if needed
    Tensor padded_input = (padding_ > 0) ? applyPadding(input) : input;

    // Extract input dimensions
    int C_in = input_shape[0];  // Number of input channels
    int H_in = input_shape[1];  // Input height
    int W_in = input_shape[2];  // Input width

    // Extract output dimensions
    int C_out = output_shape[0]; // Number of output channels
    int H_out = output_shape[1]; // Output height
    int W_out = output_shape[2]; // Output width

    // Create output tensor
    Tensor output(C_out, H_out, W_out);

    // Perform the convolution operation
    for (int c_out = 0; c_out < C_out; ++c_out) {
        // Initialize the output channel with biases
        Eigen::MatrixXf out_channel = Eigen::MatrixXf::Constant(H_out, W_out, biases[c_out](0));
        for (int c_in = 0; c_in < C_in; ++c_in) {
            // Extract the weight kernel for this output-input channel pair
            const Eigen::MatrixXf& kernel = weights[c_out * C_in + c_in];

            // Perform the convolution operation
            for (int i = 0; i < H_out; ++i) {
                for (int j = 0; j < W_out; ++j) {
                    // Calculate the region in the padded input
                    int start_row = i * stride_;
                    int start_col = j * stride_;

                    // Extract the region of interest
                    Eigen::MatrixXf region = padded_input[c_in].block(
                        start_row, start_col, kernel_size_, kernel_size_);

                    // Perform element-wise multiplication and sum
                    out_channel(i, j) += (region.array() * kernel.array()).sum();
                }
            }
        }

        // Assign the computed channel to the output tensor
        output[c_out] = out_channel;
    }

    return output;
}

Tensor Conv2D::backward(const Tensor& grad_output) {
    // Initialize gradient tensors
    Tensor grad_input(input_shape);
    grad_weights.clear();
    grad_biases.clear();

    // Iterate through each output channel
    for (int oc = 0; oc < output_shape[0]; ++oc) {
        Eigen::MatrixXf grad_output_channel = grad_output[oc];

        // Update biases: sum of gradients over spatial dimensions
        grad_biases[oc] = grad_output_channel.rowwise().sum();

        // Iterate through each input channel
        for (int ic = 0; ic < input_shape[0]; ++ic) {
            Eigen::MatrixXf input_channel = cache_input[ic];
            Eigen::MatrixXf kernel = weights[oc * input_shape[0] + ic];

            // Compute gradients for weights
            for (int i = 0; i < kernel.rows(); ++i) {
                for (int j = 0; j < kernel.cols(); ++j) {
                    // Extract region for convolution
                    Eigen::MatrixXf region = input_channel.block(i, j, grad_output_channel.rows(), grad_output_channel.cols());
                    grad_weights[oc * input_shape[0] + ic](i, j) += (region.array() * grad_output_channel.array()).sum();
                }
            }

            // Compute gradients for input
            Eigen::MatrixXf flipped_kernel = kernel.colwise().reverse().rowwise().reverse();
            Eigen::MatrixXf grad_input_channel = Eigen::MatrixXf::Zero(input_shape[1], input_shape[2]);

            for (int i = 0; i < input_shape[1]; ++i) {
                for (int j = 0; j < input_shape[2]; ++j) {
                    // Calculate region in grad_output_channel
                    int start_row = i - padding_;
                    int start_col = j - padding_;
                    int end_row = start_row + kernel_size_;
                    int end_col = start_col + kernel_size_;

                    // Perform boundary checks
                    if (start_row >= 0 && start_col >= 0 && end_row <= grad_output_channel.rows() && end_col <= grad_output_channel.cols()) {
                        Eigen::MatrixXf region = grad_output_channel.block(start_row, start_col, kernel_size_, kernel_size_);
                        grad_input_channel(i, j) += (region.array() * flipped_kernel.array()).sum();
                    }
                }
            }

            // Accumulate gradient for the current input channel
            grad_input[ic] += grad_input_channel;
        }
    }


    // Return the computed gradient tensor
    return grad_input;
}