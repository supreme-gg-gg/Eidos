#include "../include/layers/regularization.h"
#include "../include/tensor.hpp"
#include <Eigen/Dense>
#include "../include/console.hpp"

Dropout::Dropout(float probability) : probability(probability) {}

Tensor Dropout::forward(const Tensor& input) {

    Eigen::MatrixXf input_mat = input.getSingleMatrix();

    if (!training) {
        return input; // No dropout during inference
    }
    // Generate random mask and apply dropout
    mask = (Eigen::MatrixXf::Random(input_mat.rows(), input_mat.cols()).array() > probability).cast<float>();
    return Tensor(input_mat.array() * mask.array() / (1.0f - probability));
}

Tensor Dropout::backward(const Tensor& grad_output) {
    // Backprop through only the active nodes
    return Tensor(mask.array() * grad_output.getSingleMatrix().array());
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

Tensor BatchNorm::forward(const Tensor& input) {

    Eigen::MatrixXf input_mat = input.getSingleMatrix();

    if (input_mat.rows() == 1) {
        Console::log("Using BatchNorm with a single sample. This will cause unintended behaviour. Consider using LayerNorm instead.", Console::WARNING);
        Console::log("George is mad at such a small batch size. He's a Georgist, after all.", Console::WORSHIP);
    }

    // Compute mean across columns (features)
    mean = input_mat.colwise().mean();
    
    // Center the input
    centered_input = input_mat.rowwise() - mean.transpose();
    
    // Compute variance
    variance = (centered_input.array().square().colwise().sum()) / (input_mat.rows() - 1 + epsilon);
    
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
    
    return Tensor(output);
}

Tensor BatchNorm::backward(const Tensor& grad_output) {

    Eigen::MatrixXf grad_output_mat = grad_output.getSingleMatrix();

    int m = grad_output_mat.rows();
    
    // Compute gradients for gamma and beta
    grad_gamma = (grad_output_mat.array() * normalized_input.array()).colwise().sum();
    grad_beta = grad_output_mat.colwise().sum();
    
    // Compute gradient with respect to normalized input
    Eigen::MatrixXf grad_normalized = grad_output_mat.array().rowwise() * gamma.row(0).array();
    
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
    
    return Tensor(grad_input);
}

void BatchNorm::serialize(std::ofstream& toFileStream) const {
    toFileStream.write((char*)&num_features, sizeof(int));
    toFileStream.write((char*)&epsilon, sizeof(float));
    toFileStream.write((char*)gamma.data(), num_features * sizeof(float));
    toFileStream.write((char*)beta.data(), num_features * sizeof(float));
    toFileStream.write((char*)running_mean.data(), num_features * sizeof(float));
    toFileStream.write((char*)running_variance.data(), num_features * sizeof(float));
}

BatchNorm* BatchNorm::deserialize(std::ifstream& fromFileStream) {
    int num_features;
    float epsilon;
    fromFileStream.read((char*)&num_features, sizeof(int));
    fromFileStream.read((char*)&epsilon, sizeof(float));
    BatchNorm* layer = new BatchNorm(num_features, epsilon);
    fromFileStream.read((char*)layer->get_weights()[0]->data(), num_features * sizeof(float));
    fromFileStream.read((char*)layer->get_bias()[0]->data(), num_features * sizeof(float));
    fromFileStream.read((char*)layer->get_running_mean()[0]->data(), num_features * sizeof(float));
    fromFileStream.read((char*)layer->get_running_variance()[0]->data(), num_features * sizeof(float));
}