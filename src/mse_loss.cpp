#include "../include/mse_loss.h"
#include <Eigen/Dense>

float MSELoss::forward(const Eigen::MatrixXf& predictions, const Eigen::MatrixXf& targets) {
    // Mean squared error: 1/n * sum((y_pred - y_true)^2)
    this->predictions = predictions;
    this->targets = targets;
    Eigen::MatrixXf diff = predictions - targets;
    return (diff.array().square().sum()) / predictions.rows();
}

Eigen::MatrixXf MSELoss::backward() const {
    // Derivative of MSE = 2 * (pred - true) / n
    return 2 * (predictions - targets) / predictions.rows();
}