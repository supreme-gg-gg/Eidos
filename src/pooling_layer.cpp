#include "../include/layers/pooling_layer.h"
#include "../include/tensor.hpp"
#include <Eigen/Dense>

MaxPooling2D::MaxPooling2D(int pool_size, int stride) : pool_size(pool_size), stride(stride) {}

Tensor MaxPooling2D::forward(const Tensor& input) {
    this->input = input;
    std::tuple<int, int, int> shape = input.shape();
    int channels = std::get<0>(shape);
    int height = std::get<1>(shape);
    int width = std::get<2>(shape);
    int output_height = (height - pool_size) / stride + 1;
    int output_width = (width - pool_size) / stride + 1;
    Tensor output = Tensor(channels, output_height, output_width);
    this->mask = Tensor(channels, output_height, output_width);

    // iterate over each channel independently
    for (int c = 0; c < channels; c++) {
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

    return output;
}

Tensor MaxPooling2D::backward(const Tensor& grad_output) {
    std::tuple<int, int, int> shape = grad_output.shape();
    int channels = std::get<0>(shape);
    int height = std::get<1>(shape);
    int width = std::get<2>(shape);
    Tensor grad_input = Tensor(channels, height * stride, width * stride);

    // iterate over each channel independently
    for (int c = 0; c < channels; c++) {
        // Get gradient and mask for current channel
        Eigen::MatrixXf grad_channel = grad_output[c];
        Eigen::MatrixXf mask_channel = this->mask[c];

        // Unravel the mask to get the indices of the max values
        for (int i = 0; i < height; i++) {
            for (int j = 0; j < width; j++) {
                int max_idx = mask_channel(i, j);
                int max_i = max_idx / pool_size;
                int max_j = max_idx % pool_size;

                // Scatter the gradient to the corresponding location in the input gradient matrix
                grad_input(c, i * stride + max_i, j * stride + max_j) += grad_channel(i, j);
            }
        }
    }

    return grad_input;
}

AveragePooling2D::AveragePooling2D(int pool_size, int stride) : pool_size(pool_size), stride(stride) {}

Tensor AveragePooling2D::forward(const Tensor& input) {
    this->input = input;
    std::tuple<int, int, int> shape = input.shape();
    int channels = std::get<0>(shape);
    int height = std::get<1>(shape);
    int width = std::get<2>(shape);
    int output_height = (height - pool_size) / stride + 1;
    int output_width = (width - pool_size) / stride + 1;
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
    std::tuple<int, int, int> shape = grad_output.shape();
    int channels = std::get<0>(shape);
    int height = std::get<1>(shape);
    int width = std::get<2>(shape);
    Tensor grad_input = Tensor(channels, height * stride, width * stride);

    // iterate over each channel independently
    for (int c = 0; c < channels; c++) {
        Eigen::MatrixXf grad_channel = grad_output[c];
        for (int i = 0; i < height; i++) {
            for (int j = 0; j < width; j++) {
                grad_input(c, i * stride, j * stride) += grad_channel(i, j) / (pool_size * pool_size);
            }
        }
    }

    return grad_input;
}