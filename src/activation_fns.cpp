#include "../include/activation_fns.h"
#include <Eigen/Dense>
#include <iostream>

Eigen::MatrixXf ReLU::forward(const Eigen::MatrixXf& input) {
    cache_output = (input.array() > 0).cast<float>(); // Cache the mask
    return input.cwiseMax(0);                        // Apply ReLU activation
}

Eigen::MatrixXf ReLU::backward(const Eigen::MatrixXf& grad_output) {
    return grad_output.array() * cache_output.array(); // Apply the cached mask to gradients
}

Eigen::MatrixXf LeakyReLU::forward(const Eigen::MatrixXf& input) {
    cache_output = (input.array() > 0).cast<float>() + alpha * (input.array() <= 0).cast<float>();
    return input.cwiseMax(0) + alpha * input.cwiseMin(0); // Leaky ReLU activation
}

Eigen::MatrixXf LeakyReLU::backward(const Eigen::MatrixXf& grad_output) {
    return grad_output.array() * cache_output.array(); // Apply the cached mask to gradients
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
    // Compute the exponentials in a numerically stable way
    Eigen::MatrixXf exp_logits = (logits.array().rowwise() - logits.colwise().maxCoeff().array()).exp();
    Eigen::VectorXf row_sums = exp_logits.array().rowwise().sum();

    float epsilon = 1e-10f;  // Small constant to avoid division by zero
    cache_output = exp_logits.array().colwise() / (row_sums.array() + epsilon);

    return cache_output;
}

Eigen::MatrixXf Softmax::backward(const Eigen::MatrixXf& grad_output) {
    Eigen::MatrixXf grad = Eigen::MatrixXf::Zero(grad_output.rows(), grad_output.cols());
    for (int i = 0; i < grad.rows(); ++i) {
        Eigen::RowVectorXf y = cache_output.row(i);  // Softmax output for the current sample
        float dot = grad_output.row(i).dot(y);      // Inner product
        grad.row(i) = grad_output.row(i).cwiseProduct(y) - y * dot;
    }
    return grad;
}

Eigen::MatrixXf Tanh::forward(const Eigen::MatrixXf& input) {
    cache_output = input.array().tanh();
    return cache_output;
}

Eigen::MatrixXf Tanh::backward(const Eigen::MatrixXf& grad_output) {
    return grad_output.array() * (1 - cache_output.array().square());
}