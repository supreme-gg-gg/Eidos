#include "../include/layers/flatten_layer.h"
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

Eigen::MatrixXf FlattenLayer::forward(const std::vector<Eigen::MatrixXf>& input) {
    if (input.size() != input_shape[0]) {
        throw std::invalid_argument("Input tensor size does not match expected Channels.");
    }

    int batch_size = input[0].rows();
    int flattened_size = this->output_shape[1];
    Eigen::MatrixXf output(batch_size, flattened_size);

    for (int i = 0; i < this->output_shape[0]; ++i) {
        Eigen::VectorXf flattened(flattened_size);
        int offset = 0;
        for (int channel = 0; channel < input.size(); ++channel) {
            flattened.segment(offset, input[channel].cols()) = input[channel].row(i);
            offset += input[channel].cols();
        }
        output.row(i) = flattened.transpose();
    }
    return output;
}

std::vector<Eigen::MatrixXf> FlattenLayer::backward(const Eigen::MatrixXf& grad, bool flag=false) {
    int batch_size = grad.rows();
    int flattened_size = grad.cols();
    if (flattened_size != output_shape[1]) {
        throw std::invalid_argument("Gradient size does not match expected flattened size.");
    }

    std::vector<Eigen::MatrixXf> reshaped_grad(input_shape[0], Eigen::MatrixXf::Zero(batch_size, input_shape[1] * input_shape[2]));

    for (int i = 0; i < batch_size; ++i) {
        int offset = 0;
        for (int channel = 0; channel < input_shape[0]; ++channel) {
            reshaped_grad[channel].row(i) = grad.row(i).segment(offset, input_shape[1] * input_shape[2]);
            offset += input_shape[1] * input_shape[2];
        }
    }
    return reshaped_grad;
}