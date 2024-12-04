#include "../include/mse_loss.h"
#include <Eigen/Dense>

float MSELoss::compute_loss(const Eigen::MatrixXf& predictions, const Eigen::MatrixXf& targets) const {
    // Mean squared error: 1/n * sum((y_pred - y_true)^2)
    Eigen::MatrixXf diff = predictions - targets;
    return (diff.array().square().sum()) / predictions.rows();
}

Eigen::MatrixXf MSELoss::compute_gradient(const Eigen::MatrixXf& predictions, const Eigen::MatrixXf& targets) const {
    // Derivative of MSE = 2 * (pred - true) / n
    return 2 * (predictions - targets) / predictions.rows();
}