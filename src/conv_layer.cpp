#include "../include/layers/conv_layer.h"
#include "../include/tensor.hpp"
#include <random>
#include <Eigen/Dense>

ConvolutionalLayer::ConvolutionalLayer(int input_channels, int output_channels, 
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

std::vector<int> ConvolutionalLayer::calculateOutputShape() const {
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

Tensor ConvolutionalLayer::forward(const Tensor& input) {
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

Tensor ConvolutionalLayer::backward(const Tensor& grad_output) {
    // Extract the gradient of the output as a matrix for further operations
    Eigen::MatrixXf grad_output_matrix = grad_output.getSingleMatrix();

    // Get the input shape from the cache_input tensor (it has already been set in the forward pass)
    std::vector<int> input_shape = cache_input.shape();
    int input_channels = input_shape[0];
    int input_height = input_shape[1];
    int input_width = input_shape[2];
    
    // Initialize the gradient of the input (this will have the same shape as the input)
    Tensor grad_input(input_shape); // This tensor will hold the gradients for the input

    // Initialize the gradient of weights and biases (for updating during backprop)
    grad_weights.clear();
    grad_biases.clear();
    
    // Loop over each filter (output channel)
    for (int out_c = 0; out_c < grad_output.shape()[0]; ++out_c) {
        for (int in_c = 0; in_c < input_channels; ++in_c) {
            // Loop over the spatial dimensions (height and width) of the input and output
            for (int i = 0; i < input_height; ++i) {
                for (int j = 0; j < input_width; ++j) {
                    // Here you should apply the kernel convolution to calculate the gradients
                    // For the input gradient, we need to perform a "reverse" convolution.

                    // Apply the filter (convolution)
                    for (int k = 0; k < kernel_size_; ++k) {
                        for (int l = 0; l < kernel_size_; ++l) {
                            // We compute gradients for each input channel and spatial position
                            int grad_x = i - padding_ + k;  // Reverse convolution index
                            int grad_y = j - padding_ + l;  // Reverse convolution index

                            // Here grad_input and grad_output tensors are indexed correctly by their dimensions
                            grad_input[in_c](grad_x, grad_y) += grad_output[out_c](i, j) * weights[out_c](k, l);
                        }
                    }
                }
            }
        }
    }

    // Now, let's compute the gradients with respect to weights and biases
    for (int out_c = 0; out_c < grad_output.shape()[0]; ++out_c) {
        for (int in_c = 0; in_c < input_channels; ++in_c) {
            // Calculate gradients for weights and biases using the chain rule
            for (int i = 0; i < kernel_size_; ++i) {
                for (int j = 0; j < kernel_size_; ++j) {
                    grad_weights[out_c](i, j) += grad_output[out_c](i, j);
                }
            }
            grad_biases[out_c] += grad_output[out_c];  // Gradient w.r.t. biases
        }
    }

    // Return the calculated gradient w.r.t. input as a tensor
    return grad_input;
}