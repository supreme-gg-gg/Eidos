#include "../include/dense_layer.h"
#include <iostream>

DenseLayer::DenseLayer(int input_size, int output_size, Activation* activation_func)
    : activation(activation_func) {

    // TODO: Xavier initialization
    weights = Eigen::MatrixXf::Random(output_size, input_size);
    bias = Eigen::VectorXf::Zero(output_size, 1);

    // Used to store gradients
    grad_weights = Eigen::MatrixXf::Zero(output_size, input_size);
    grad_bias = Eigen::VectorXf::Zero(output_size, 1);
}

Eigen::MatrixXf DenseLayer::forward(const Eigen::MatrixXf& input) {
    if (input.rows() == 0 || input.cols() == 0) {
        throw std::invalid_argument("Input matrix cannot be empty.");
    }

    this->input = input;
    // Broadcast bias to match the number of samples
    output = (weights * input) + bias.replicate(1, input.cols()); // Linear transformation
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
    Eigen::VectorXf grad_bias = grad_output.rowwise().sum();

    return grad_input;
}

Eigen::MatrixXf DenseLayer::get_weights_gradient() const {
    return grad_weights;
}

Eigen::VectorXf DenseLayer::get_bias_gradient() const {
    return grad_bias;
}