#include "include/dense_layer.h"
#include <iostream>

DenseLayer::DenseLayer(int input_size, int output_size, Activation* activation_func)
    : activation(activation_func) {
    weights = Eigen::MatrixXf::Random(output_size, input_size);
    bias = Eigen::MatrixXf::Zero(output_size, 1);

    grad_weights = Eigen::MatrixXf::Zero(output_size, input_size);
    grad_bias = Eigen::MatrixXf::Zero(output_size, 1);
}

Eigen::MatrixXf DenseLayer::forward(const Eigen::MatrixXf& input) {
    if (input.rows() == 0 || input.cols() == 0) {
        throw std::invalid_argument("Input matrix cannot be empty.");
    }

    this->input = input;
    output = (weights * input).colwise() + bias; // Linear transformation
    return activation->forward(output); // Apply activation function

    // TODO: Implement batch normalization
}

Eigen::MatrixXf DenseLayer::backward(const Eigen::MatrixXf& grad_output) {

    // Backprop through activation
    Eigen::MatrixXf grad_activation = activation->backward(grad_output);

    // Gradient w.r.t. input
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