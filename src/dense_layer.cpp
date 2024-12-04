#include "include/dense_layer.h"
#include <iostream>

DenseLayer::DenseLayer(int input_size, int output_size) {
    weights = Eigen::MatrixXf::Random(output_size, input_size);
    bias = Eigen::MatrixXf::Zero(output_size, 1);

    grad_weights = Eigen::MatrixXf::Zero(output_size, input_size);
    grad_bias = Eigen::MatrixXf::Zero(output_size, 1);
}

Eigen::MatrixXf DenseLayer::forward(const Eigen::MatrixXf& input) {
    this->input = input;
    return (weights * input).colwise() + bias;
}

Eigen::MatrixXf DenseLayer::backward(const Eigen::MatrixXf& grad_output) {
    // Calculate the gradient of the loss with respect to the input
    Eigen::MatrixXf grad_input = weights.transpose() * grad_output;

    // Gradeitn w.r.t. weights and biases
    Eigen::MatrixXf grad_weights = grad_output * input.transpose();
    Eigen::MatrixXf grad_bias = grad_output.rowwise().sum();

    return grad_input;
}

Eigen::MatrixXf DenseLayer::get_weights_gradient() const {
    return grad_weights;
}

Eigen::MatrixXf DenseLayer::get_bias_gradient() const {
    return grad_bias;
}