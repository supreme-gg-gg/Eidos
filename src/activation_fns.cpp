#include "../include/activation_fns.h"
#include <Eigen/Dense>
#include <iostream>

Eigen::MatrixXf ReLU::forward(const Eigen::MatrixXf& input) {
    cache_output = (input.array() > 0).cast<float>();
    return input.cwiseMax(0); // ReLU activation
}

Eigen::MatrixXf ReLU::backward(const Eigen::MatrixXf& grad_output) {
    return grad_output.cwiseProduct(cache_output); // Grad of ReLU
}

Eigen::MatrixXf Sigmoid::forward(const Eigen::MatrixXf& input) {
    cache_output = 1.0f / (1.0f + (-input.array()).exp());
    return cache_output;
}

Eigen::MatrixXf Sigmoid::backward(const Eigen::MatrixXf& grad_output) {
    Eigen::MatrixXf sigmoid_grad = cache_output.array() * (1.0f - cache_output.array());
    return grad_output.array() * sigmoid_grad.array();
}

Eigen::MatrixXf Softmax::forward(const Eigen::MatrixXf& logits) {
    Eigen::MatrixXf exp_logits = (logits.array() - logits.rowwise().maxCoeff().array()).exp(); // Subtract max for numerical stability
    Eigen::MatrixXf probabilities = exp_logits.array().colwise() / exp_logits.rowwise().sum().array(); // Normalize by sum
    cache_output = probabilities; // Cache the output for backpropagation
    return probabilities;
}

Eigen::MatrixXf Softmax::backward(const Eigen::MatrixXf& grad_output) {
    Eigen::MatrixXf grad = cache_output;
    for (int i = 0; i < grad.rows(); ++i) {
        Eigen::MatrixXf jacobian = grad.row(i).transpose() * grad.row(i); // Jacobian
        jacobian.diagonal() -= grad.row(i).transpose(); // subtract diagonal
        grad.row(i) = grad_output.row(i) * jacobian; // chain rule
    }
    return grad;
}
