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

float CategoricalCrossEntropyLoss::forward(const Eigen::MatrixXf& predictions, const Eigen::MatrixXf& targets) {
    // Clip predictions to avoid log(0) for numerical stability
    this->predictions = predictions;
    this->targets = targets;
    Eigen::MatrixXf clipped_preds = predictions.cwiseMax(1e-7f).cwiseMin(1.0f - 1e-7f);
    // CrossEntropyLoss = - sum(targets * log(predictions)) (element-wise sum)
    return - (targets.array() * clipped_preds.array().log()).sum() / predictions.rows();  // Use predictions.rows() for consistency
}

Eigen::MatrixXf CategoricalCrossEntropyLoss::backward() const {
    // Avoid divide by zero in log
    Eigen::MatrixXf safe_predictions = predictions.cwiseMax(1e-7f);  // Ensure clipping is applied to predictions

    // Compute gradient element-wise
    Eigen::MatrixXf gradient = -targets.cwiseQuotient(safe_predictions);

    // Normalize by batch size (N)
    return gradient / static_cast<float>(predictions.rows());  // Consistent use of predictions.rows()
}

// Forward pass: Calculate loss for binary classification
float BinaryCrossEntropyLoss::forward(const Eigen::MatrixXf& predictions, const Eigen::MatrixXf& targets) {
    this->predictions = predictions;
    this->targets = targets;

    // Clip predictions to avoid log(0) for numerical stability
    Eigen::MatrixXf clipped_preds = predictions.cwiseMax(1e-7f).cwiseMin(1.0f - 1e-7f);

    // CrossEntropyLoss = - sum(targets * log(predictions) + (1 - targets) * log(1 - predictions))
    return - (targets.array() * clipped_preds.array().log() + (1 - targets.array()) * (1 - clipped_preds.array()).log()).sum() / targets.rows();
}

Eigen::MatrixXf BinaryCrossEntropyLoss::backward() const {
    // Avoid divide by zero in log
    Eigen::MatrixXf safe_predictions = predictions.cwiseMax(1e-7f);

    // Compute gradient for binary cross entropy
    Eigen::MatrixXf gradient = (safe_predictions - targets).cwiseQuotient(safe_predictions.cwiseProduct(Eigen::MatrixXf::Constant(safe_predictions.rows(), safe_predictions.cols(), 1.0f) - safe_predictions));

    // Normalize by batch size (N)
    return gradient / static_cast<float>(predictions.rows());
}