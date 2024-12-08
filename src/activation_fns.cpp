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

Eigen::MatrixXf LeakyReLU::forward(const Eigen::MatrixXf& input) {
    cache_output = (input.array() > 0).cast<float>() + alpha * (input.array() <= 0).cast<float>();
    return input.cwiseMax(0) + alpha * input.cwiseMin(0); // Leaky ReLU activation
}

Eigen::MatrixXf LeakyReLU::backward(const Eigen::MatrixXf& grad_output) {
    return grad_output.cwiseProduct(cache_output); // Grad of Leaky ReLU
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

    // Normalize by the sum of exponentials in each row, adding epsilon to avoid division by zero
    Eigen::VectorXf row_sums = exp_logits.colwise().sum();
    float epsilon = 1e-10f;  // Small constant to avoid division by zero
    cache_output = exp_logits.array().rowwise() / (row_sums.transpose().array() + epsilon);

    return cache_output;
}

Eigen::MatrixXf Softmax::backward(const Eigen::MatrixXf& grad_output) {
    Eigen::MatrixXf grad = Eigen::MatrixXf::Zero(grad_output.rows(), grad_output.cols());

    for (int i = 0; i < grad.rows(); ++i) {
        // Extract the softmax output for the current sample
        Eigen::RowVectorXf y = cache_output.row(i);

        // Compute the Jacobian matrix for softmax (dy/dx)
        Eigen::MatrixXf jacobian = Eigen::MatrixXf::Identity(y.size(), y.size()) - y.transpose() * y;

        // Apply chain rule to compute the gradient for the current sample
        grad.row(i) = grad_output.row(i) * jacobian;
    }

    return grad;
}

Eigen::Matrix Tanh::forward(const Eigen::Matrix& input) {
    cache_output = input.array().tanh();
    return cache_output;
}

Eigen::Matrix Tanh::backward(const Eigen::Matrix& grad_output) {
    return grad_output.array() * (1 - cache_output.array().square());
}