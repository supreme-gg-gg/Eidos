#include "../include/loss_fns.h"
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

// TODO: Fix cross entropy
float CrossEntropyLoss::forward(const Eigen::MatrixXf& predictions, const Eigen::MatrixXf& targets) {
    // Clip predictions to avoid log(0) for numerical stability
    Eigen::MatrixXf clipped_preds = predictions.cwiseMax(1e-7f).cwiseMin(1.0f - 1e-7f);

    // CrossEntropyLoss = - sum(targets * log(predictions)) (element-wise sum)
    return - (targets.array() * clipped_preds.array().log()).sum() / targets.rows();
}

Eigen::MatrixXf CrossEntropyLoss::backward() const {
    // Avoid divide by zero in log
    Eigen::MatrixXf safe_predictions = predictions.cwiseMax(1e-7); 

    // Compute gradient element-wise
    Eigen::MatrixXf gradient = -targets.cwiseQuotient(safe_predictions);

    // Normalize by batch size (N)
    return gradient / static_cast<float>(predictions.rows());
}