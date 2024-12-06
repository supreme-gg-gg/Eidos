#include "../include/dense_layer.h"

DenseLayer::DenseLayer(int input_size, int output_size)
    : weights(Eigen::MatrixXf::Random(output_size, input_size)),
      bias(Eigen::VectorXf::Zero(output_size)),
      grad_weights(Eigen::MatrixXf::Zero(output_size, input_size)),
      grad_bias(Eigen::VectorXf::Zero(output_size)) {}

Eigen::MatrixXf DenseLayer::forward(const Eigen::MatrixXf& input) {
    this->input = input;
    return (weights * input).colwise() + bias;
}

Eigen::MatrixXf DenseLayer::backward(const Eigen::MatrixXf& grad_output) {
    grad_weights = grad_output * input.transpose();
    grad_bias = grad_output.rowwise().sum();
    return weights.transpose() * grad_output;
}

bool DenseLayer::has_weights() const { return true; }

bool DenseLayer::has_bias() const { return true; }

Eigen::MatrixXf* DenseLayer::get_weights() { return &weights; }

Eigen::MatrixXf* DenseLayer::get_grad_weights() { return &grad_weights; }

Eigen::VectorXf* DenseLayer::get_bias() { return &bias; }

Eigen::VectorXf* DenseLayer::get_grad_bias() { return &grad_bias; }