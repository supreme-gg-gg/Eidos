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

// TODO: Fix the softmax implementation
Eigen::MatrixXf Softmax::forward(const Eigen::MatrixXf& logits) {
    // Subtract the maximum value in each column for numerical stability
    Eigen::VectorXf max_per_col = logits.colwise().maxCoeff(); // Column vector of column max values
    
    // Create a matrix of max values expanded to match logits
    Eigen::MatrixXf max_matrix = max_per_col.transpose().replicate(logits.rows(), 1);
    
    // Subtract max values from each column
    Eigen::MatrixXf shifted_logits = logits - max_matrix;
    
    // Compute exponentials
    Eigen::MatrixXf exp_logits = shifted_logits.array().exp();
    
    // Compute column-wise sums
    Eigen::VectorXf col_sums = exp_logits.colwise().sum();
    
    // Normalize by dividing each column by its sum
    Eigen::MatrixXf probabilities = exp_logits.array().colwise() / col_sums.array();
    
    // Cache the output for backpropagation
    cache_output = probabilities;
    return probabilities;
}

Eigen::MatrixXf Softmax::backward(const Eigen::MatrixXf& grad_output) {
    Eigen::MatrixXf grad = Eigen::MatrixXf::Zero(grad_output.rows(), grad_output.cols());

    for (int i = 0; i < grad.rows(); ++i) {
        // Extract the softmax output for the current sample
        Eigen::RowVectorXf y = cache_output.row(i);

        // Compute the Jacobian matrix for softmax (dy/dx)
        Eigen::MatrixXf jacobian = y.transpose() * (Eigen::MatrixXf::Identity(y.size(), y.size()) - y);

        // Apply chain rule to compute the gradient for the current sample
        grad.row(i) = grad_output.row(i) * jacobian;
    }

    return grad;
}
