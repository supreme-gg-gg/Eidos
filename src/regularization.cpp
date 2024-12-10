#include "../include/regularization.h"
#include <Eigen/Dense>

Eigen::MatrixXf Dropout::forward(const Eigen::MatrixXf& input) {

    if (!training) {
            return input; // No dropout during inference
        }
        // Generate random mask and apply dropout
        mask = (Eigen::MatrixXf::Random(input.rows(), input.cols()).array() > probability).cast<float>();
        return input.array() * mask.array() / (1.0f - probability);
}

Eigen::MatrixXf Dropout::backward(const Eigen::MatrixXf& grad_output) {
    // Backprop through only the active nodes
    return mask.array() * grad_output.array();
}

BatchNorm::BatchNorm(int num_features, float epsilon)
    : epsilon(epsilon), num_features(num_features), training(true) {
    this->epsilon = epsilon;
    gamma = Eigen::MatrixXf::Ones(1, num_features);
    beta = Eigen::MatrixXf::Zero(1, num_features);
    running_mean = Eigen::MatrixXf::Zero(1, num_features);
    running_variance = Eigen::MatrixXf::Zero(1, num_features);
}

Eigen::MatrixXf BatchNorm::forward(const Eigen::MatrixXf& input) {
    if (training) {
        // Compute mean and variance
        mean = input.rowwise().mean();
        Eigen::MatrixXf centered = input.rowwise() - mean;  // Center the input (subtract mean)
        variance = (centered.array().square().rowwise().sum()) / input.rows();  // Compute variance
        // Update running mean and variance
        running_mean = 0.9 * running_mean + 0.1 * mean;
        running_variance = 0.9 * running_variance + 0.1 * variance;
        
        // Store centered input for the backward pass
        centered_input = centered;
    } else {
        // Use running averages during inference
        mean = running_mean;
        variance = running_variance;
    }

    // Normalize input
    normalized_input = centered_input.array() / (variance.array() + epsilon).sqrt();  // Normalize centered input
    // Scale and shift
    return gamma.array() * normalized_input.array() + beta.array();  // Apply scaling (gamma) and shifting (beta)
}

Eigen::MatrixXf BatchNorm::backward(const Eigen::MatrixXf& grad_output) {
    int m = grad_output.rows();

    // Compute gradients for gamma and beta
    grad_gamma = (grad_output.array() * normalized_input.array()).rowwise().sum();
    grad_beta = grad_output.rowwise().sum();

    // Compute gradient with respect to normalized input
    Eigen::MatrixXf grad_normalized = grad_output.array() * gamma.array();

    // Compute gradient with respect to variance and mean
    Eigen::MatrixXf grad_variance = (grad_normalized.array() * centered_input.array()).rowwise().sum() * -0.5f *
                                    (variance.array() + epsilon).pow(-1.5);
    Eigen::MatrixXf grad_mean = grad_normalized.rowwise().sum() * -1.0f / (variance.array() + epsilon).sqrt() +
                                    grad_variance.array() * -2.0f * centered_input.array().rowwise().sum() / m;

    // Compute the gradient with respect to the original input
    Eigen::MatrixXf grad_input = grad_normalized.array() / (variance.array() + epsilon).sqrt() +
                                    grad_variance.array() * 2.0f * centered_input.array() / m +
                                    grad_mean.array() / m;

    return grad_input;
}