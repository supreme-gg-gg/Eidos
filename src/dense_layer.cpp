#include "../include/dense_layer.h"

DenseLayer::DenseLayer(int input_size, int output_size)
    : weights(Eigen::MatrixXf::Random(input_size, output_size)), // input -> output
      bias(Eigen::VectorXf::Zero(output_size)),
      grad_weights(Eigen::MatrixXf::Zero(input_size, output_size)),
      grad_bias(Eigen::VectorXf::Zero(output_size)) {}

Eigen::MatrixXf DenseLayer::forward(const Eigen::MatrixXf& input) {
    this->input = input;
    return (input * weights).rowwise() + bias.transpose(); // Row wise bias addition
}

Eigen::MatrixXf DenseLayer::backward(const Eigen::MatrixXf& grad_output) {
    grad_weights = input.transpose() * grad_output; // dL/dW = X^T * dL/dY
    grad_bias = grad_output.colwise().sum(); // dL/db = sum(dL/dY)
    return grad_output * weights.transpose(); // dL/dX = dL/dY * W^T
}

bool DenseLayer::has_weights() const { return true; }

bool DenseLayer::has_bias() const { return true; }

Eigen::MatrixXf* DenseLayer::get_weights() { return &weights; }

Eigen::MatrixXf* DenseLayer::get_grad_weights() { return &grad_weights; }

Eigen::VectorXf* DenseLayer::get_bias() { return &bias; }

Eigen::VectorXf* DenseLayer::get_grad_bias() { return &grad_bias; }