#include "../include/Eidos/layers/pooling_layer.h"
#include "../include/Eidos/tensor.hpp"
#include <Eigen/Dense>
#include <thread>
#include <mutex>

MaxPooling2D::MaxPooling2D(int pool_size, int stride) : pool_size(pool_size), stride(stride) {}

Tensor MaxPooling2D::forward(const Tensor& input) {
    // Initialize the input tensor and shapes
    this->input = input;
    std::tuple<int, int, int> shape = input.shape();
    int channels = std::get<0>(shape);
    int height = std::get<1>(shape);
    int width = std::get<2>(shape);
    this->input_shape = {channels, height, width};
    int output_height = (height - pool_size) / stride + 1;
    int output_width = (width - pool_size) / stride + 1;
    this->output_shape = {channels, output_height, output_width};
    Tensor output = Tensor(channels, output_height, output_width);
    this->mask = Tensor(channels, output_height, output_width);

    std::vector<std::thread> threads;
    int num_threads = std::thread::hardware_concurrency();
    for (int thd = 0; thd < num_threads; ++thd) {
        threads.emplace_back([&](int thd) {
            // iterate over each channel independently
            for (int c = thd; c < channels; c += num_threads) {
                Eigen::MatrixXf channel = input[c];
                // Perform max pooling using valid convolutions
                for (int i = 0; i < output_height; i++) {
                    for (int j = 0; j < output_width; j++) {
                        Eigen::MatrixXf window = channel.block(i * stride, j * stride, pool_size, pool_size);
                        int max_idx;
                        float max_val = window.reshaped().maxCoeff(&max_idx);
                        output(c, i, j) = max_val;
                        this->mask(c, i, j) = max_idx;
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

Tensor MaxPooling2D::backward(const Tensor& grad_output) {
    Tensor grad_input = Tensor(input_shape[0], input_shape[1], input_shape[2]);

    std::vector<std::thread> threads;
    int num_threads = std::thread::hardware_concurrency();
    for (int thd = 0; thd < num_threads; ++thd) {
        threads.emplace_back([&](int thd) {
            // iterate over each channel independently
            for (int c = thd; c < output_shape[0]; c += num_threads) { // channel
                // Get gradient and mask for current channel
                Eigen::MatrixXf grad_channel = grad_output[c];
                Eigen::MatrixXf mask_channel = this->mask[c];

                // Unravel the mask to get the indices of the max values
                for (int i = 0; i < output_shape[1]; i++) { // height
                    for (int j = 0; j < output_shape[2]; j++) { // width
                        int max_idx = mask_channel(i, j);
                        int max_i = max_idx / pool_size;
                        int max_j = max_idx % pool_size;

                        // Scatter the gradient to the corresponding location in the input gradient matrix
                        grad_input(c, i * stride + max_i, j * stride + max_j) += grad_channel(i, j);
                    }
                }
            }
        }, thd);
    }

    // Wait for all threads to finish
    for (auto& thd : threads) {
        thd.join();
    }
    
    return grad_input;
}

void MaxPooling2D::serialize(std::ofstream& toFileStream) const {
    toFileStream.write((char*)&pool_size, sizeof(int));
    toFileStream.write((char*)&stride, sizeof(int));
}

MaxPooling2D* MaxPooling2D::deserialize(std::ifstream& fromFileStream) {
    int pool_size, stride;
    fromFileStream.read((char*)&pool_size, sizeof(int));
    fromFileStream.read((char*)&stride, sizeof(int));
    return new MaxPooling2D(pool_size, stride);
}

AveragePooling2D::AveragePooling2D(int pool_size, int stride) : pool_size(pool_size), stride(stride) {}

Tensor AveragePooling2D::forward(const Tensor& input) {
    this->input = input;
    std::tuple<int, int, int> shape = input.shape();
    int channels = std::get<0>(shape);
    int height = std::get<1>(shape);
    int width = std::get<2>(shape);
    this->input_shape = {channels, height, width};
    int output_height = (height - pool_size) / stride + 1;
    int output_width = (width - pool_size) / stride + 1;
    this->output_shape = {channels, output_height, output_width};
    Tensor output = Tensor(channels, output_height, output_width);

    // iterate over each channel independently
    for (int c = 0; c < channels; c++) {
        Eigen::MatrixXf channel = input[c];
        // Perform average pooling using valid convolutions
        for (int i = 0; i < output_height; i++) {
            for (int j = 0; j < output_width; j++) {
                Eigen::MatrixXf window = channel.block(i * stride, j * stride, pool_size, pool_size);
                output(c, i, j) = window.mean();
            }
        }
    }

    return output;
}

Tensor AveragePooling2D::backward(const Tensor& grad_output) {
    Tensor grad_input = Tensor(input_shape[0], input_shape[1], input_shape[2]);

    // iterate over each channel independently
    for (int c = 0; c < output_shape[0]; c++) {
        Eigen::MatrixXf grad_channel = grad_output[c];
        for (int i = 0; i < output_shape[1]; i++) {
            for (int j = 0; j < output_shape[2]; j++) {
                grad_input(c, i * stride, j * stride) += grad_channel(i, j) / (pool_size * pool_size);
            }
        }
    }

    return grad_input;
}

void AveragePooling2D::serialize(std::ofstream& toFileStream) const {
    toFileStream.write((char*)&pool_size, sizeof(int));
    toFileStream.write((char*)&stride, sizeof(int));
}

AveragePooling2D* AveragePooling2D::deserialize(std::ifstream& fromFileStream) {
    int pool_size, stride;
    fromFileStream.read((char*)&pool_size, sizeof(int));
    fromFileStream.read((char*)&stride, sizeof(int));
    return new AveragePooling2D(pool_size, stride);
}