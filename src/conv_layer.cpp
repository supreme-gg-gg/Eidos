#include "../include/Eidos/layers/conv_layer.h"
#include "../include/Eidos/tensor.hpp"
#include <random>
#include <Eigen/Dense>
#include <thread>
#include <mutex>
#include <iostream>

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

        // Initialize gradients for weights
        grad_weights[oc] = Eigen::MatrixXf::Zero(input_channels, kernel_size * kernel_size);

        // Initialize biases
        biases[oc] = Eigen::VectorXf::Zero(1); // Single bias per output channel

        // Initialize gradients for biases
        grad_biases[oc] = Eigen::VectorXf::Zero(1);
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
    
    // Create threads for parallel computation
    std::vector<std::thread> threads;
    int num_threads = std::thread::hardware_concurrency();
    for (int thd = 0; thd < num_threads; ++thd) {
        threads.emplace_back([&](int thd){
            // Each thread processes a subset of output channels
            for (int c_out = thd; c_out < C_out; c_out+=num_threads) {
                // Initialize the output channel with biases
                output[c_out] = Eigen::MatrixXf::Constant(H_out, W_out, biases[c_out](0));
                for (int c_in = 0; c_in < C_in; ++c_in) {
                    // Extract the weight kernel for this output-input channel pair
                    Eigen::MatrixXf kernel = weights[c_out].row(c_in);
                    kernel = Eigen::Map<Eigen::MatrixXf>(kernel.data(), kernel_size_, kernel_size_);

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
                            output[c_out](i, j) += (region.array() * kernel.array()).sum();
                        }
                    }
                }
            }
        }, thd);
    }
    // Wait for all threads to finish
    for (auto& thd : threads) {
        thd.join();
    }

    return output;
}

Tensor Conv2D::backward(const Tensor& grad_output) {
    // Initialize gradient tensors
    Tensor grad_input(input_shape);  // Initialize the gradient for input

    // Create threads for parallel computation
    std::vector<std::thread> threads;
    std::mutex grad_input_mutex; // mutex for protecting grad_input
    int num_threads = std::thread::hardware_concurrency();
    for (int thd = 0; thd < num_threads; ++thd) {
        threads.emplace_back([&](int thd){
            // Iterate through each output channel
            for (int c_out = thd; c_out < output_shape[0]; c_out += num_threads) {
                // Extract the gradient for the output channel
                Eigen::MatrixXf grad_output_channel = grad_output[c_out];

                // Update biases: dL/db_i = sum(dL/dz_i) for all i
                grad_biases[c_out](0) = grad_output_channel.sum();

                // Iterate through each input channel
                for (int c_in = 0; c_in < input_shape[0]; ++c_in) {

                    // Weight gradient computation using valid convolution
                    for (int i = 0; i < output_shape[1]; ++i) {
                        for (int j = 0; j < output_shape[2]; ++j) {
                            int start_row = i * stride_;
                            int start_col = j * stride_;

                            // Check if the kernel goes out of bounds
                            if (start_row + kernel_size_ > cache_input[c_in].rows() || 
                                start_col + kernel_size_ > cache_input[c_in].cols()) {
                                continue; // Skip this position
                            }

                            // Extract the valid region from the input
                            Eigen::MatrixXf region = cache_input[c_in].block(start_row, start_col, kernel_size_, kernel_size_);

                            // Accumulate gradients for weights
                            Eigen::MatrixXf region_flat = Eigen::Map<Eigen::MatrixXf>(region.data(), 1, region.size());
                            grad_weights[c_out].row(c_in) += region_flat * grad_output_channel(i, j);
                        }
                    }

                    // Extract the weight kernel for this output-input channel pair
                    Eigen::MatrixXf kernel = weights[c_out].row(c_in);
                    kernel = Eigen::Map<Eigen::MatrixXf>(kernel.data(), kernel_size_, kernel_size_);

                    // Flip the kernel 180 degrees for convolution. 
                    Eigen::MatrixXf flipped_kernel = kernel.colwise().reverse().rowwise().reverse();

                    // Compute the gradient for the input tensor using full convolution
                    Eigen::MatrixXf grad_output_padded = Eigen::MatrixXf::Zero(
                        grad_output_channel.rows() + 2 * (kernel_size_ - 1),
                        grad_output_channel.cols() + 2 * (kernel_size_ - 1)
                    );

                    grad_output_padded.block(kernel_size_ - 1, kernel_size_ - 1, 
                                            grad_output_channel.rows(), grad_output_channel.cols()) = grad_output_channel;

                    for (int i = 0; i < grad_output_channel.rows(); ++i) {
                        for (int j = 0; j < grad_output_channel.cols(); ++j) {
                            Eigen::MatrixXf region = grad_output_padded.block(i, j, kernel_size_, kernel_size_);
                            grad_input_mutex.lock(); // Lock the mutex before updating the shared resource
                            grad_input[c_in](i, j) += (region.array() * flipped_kernel.array()).sum();
                            grad_input_mutex.unlock();
                        }
                    }
                }
            }
        }, thd);
    }

    // Wait for all threads to finish
    for (auto& thd : threads) {
        thd.join();
    }

    // Return the computed gradient tensor for input
    return grad_input;
}

void Conv2D::serialize(std::ofstream& toFileStream) const {
    // Write num of input channels
    toFileStream.write((char*)&(input_shape[0]), sizeof(int));
    // Write num of output channels
    int oc = weights.size();
    toFileStream.write((char*)&oc, sizeof(int));

    // Write the layer attributes
    toFileStream.write((char*)&kernel_size_, sizeof(int));
    toFileStream.write((char*)&stride_, sizeof(int));
    toFileStream.write((char*)&padding_, sizeof(int));

    // Write the weights and biases
    for (int i = 0; i < weights.size(); ++i) {
        toFileStream.write((char*)weights[i].data(), weights[i].size() * sizeof(float));
    }
    for (int i = 0; i < biases.size(); ++i) {
        toFileStream.write((char*)biases[i].data(), biases[i].size() * sizeof(float));
    }
}

Conv2D* Conv2D::deserialize(std::ifstream& fromFileStream) {
    // Read the number of input channels
    int input_channels;
    fromFileStream.read((char*)&input_channels, sizeof(int));

    // Read the number of output channels
    int output_channels;
    fromFileStream.read((char*)&output_channels, sizeof(int));

    // Read the layer attributes
    int kernel_size, stride, padding;
    fromFileStream.read((char*)&kernel_size, sizeof(int));
    fromFileStream.read((char*)&stride, sizeof(int));
    fromFileStream.read((char*)&padding, sizeof(int));

    // Create a new Conv2D layer
    Conv2D* layer = new Conv2D(input_channels, output_channels, kernel_size, stride, padding);

    // Read the weights and biases
    for (int i = 0; i < output_channels; ++i) {
        fromFileStream.read((char*)layer->get_weights()[i]->data(), input_channels * kernel_size * kernel_size * sizeof(float));
    }
    for (int i = 0; i < output_channels; ++i) {
        fromFileStream.read((char*)layer->get_bias()[i]->data(), 1 * sizeof(float));
    }

    return layer;
}