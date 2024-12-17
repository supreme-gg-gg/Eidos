#include "../include/layers/regularization.h"
#include <Eigen/Dense>

Dropout::Dropout(float probability) : probability(probability) {}

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

void Dropout::set_training(bool training_) {
    training = training_;
}

BatchNorm::BatchNorm(int num_features, float epsilon)
    : epsilon(epsilon), num_features(num_features), training(true) {
    this->epsilon = epsilon;
    gamma = Eigen::MatrixXf::Ones(1, num_features);
    beta = Eigen::VectorXf::Zero(num_features);
    running_mean = Eigen::VectorXf::Zero(num_features);
    running_variance = Eigen::VectorXf::Zero(num_features);
}

void BatchNorm::set_training(bool training_) {
    training = training_;
}

Eigen::MatrixXf BatchNorm::forward(const Eigen::MatrixXf& input) {
    // Compute mean across columns (features)
    mean = input.colwise().mean();
    
    // Center the input
    centered_input = input.rowwise() - mean.transpose();
    
    // Compute variance
    variance = (centered_input.array().square().colwise().sum()) / (input.rows() - 1);
    
    if (training) {
        // Update running statistics
        running_mean = 0.9 * running_mean + 0.1 * mean;
        running_variance = 0.9 * running_variance + 0.1 * variance;
    } else {
        // Use running statistics during inference
        mean = running_mean;
        variance = running_variance;
    }
    
    // Normalize input
    normalized_input = centered_input.array().rowwise() / 
        (variance.array() + epsilon).sqrt().transpose();
    
    // Scale and shift 
    Eigen::MatrixXf output = (normalized_input.array().rowwise() * gamma.row(0).array()).rowwise() + beta.transpose().array();
    
    return output;
}

Eigen::MatrixXf BatchNorm::backward(const Eigen::MatrixXf& grad_output) {
    int m = grad_output.rows();
    
    // Compute gradients for gamma and beta
    grad_gamma = (grad_output.array() * normalized_input.array()).colwise().sum();
    grad_beta = grad_output.colwise().sum();
    
    // Compute gradient with respect to normalized input
    Eigen::MatrixXf grad_normalized = grad_output.array().rowwise() * gamma.row(0).array();
    
    // Compute gradient with respect to variance
    Eigen::MatrixXf grad_variance = (grad_normalized.array() * centered_input.array()).colwise().sum() * 
        -0.5f * (variance.array() + epsilon).pow(-1.5).transpose();
    
    // Compute gradient with respect to mean
    Eigen::MatrixXf grad_mean = 
        grad_normalized.colwise().sum().array() * -1.0f / (variance.array() + epsilon).sqrt().transpose() +
        grad_variance.array() * -2.0f * centered_input.colwise().sum().array() / m;
    
    // Compute the gradient with respect to the original input
    Eigen::MatrixXf grad_input = 
        grad_normalized.array().rowwise() / (variance.array() + epsilon).sqrt().transpose() +
        centered_input.array().rowwise() * grad_variance.row(0).array() * 2.0f / m + 
        (grad_mean.array() / m).replicate(m, 1);
    
    return grad_input;
}