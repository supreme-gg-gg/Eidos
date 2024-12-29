#include "../include/Eidos/loss_fns.h"
#include <Eigen/Dense>
#include <iostream>

float MSELoss::forwardMatrix(const Eigen::MatrixXf& predictions, const Eigen::MatrixXf& targets) {
    // Mean squared error: 1/n * sum((y_pred - y_true)^2)
    this->predictions = predictions;
    this->targets = targets;
    Eigen::MatrixXf diff = predictions - targets;
    return (diff.array().square().sum()) / (predictions.rows() * predictions.cols());
}

Eigen::MatrixXf MSELoss::backwardMatrix() const {
    // Derivative of MSE = 2 * (pred - true) / n
    return 2 * (predictions - targets) / predictions.rows();
}

float CategoricalCrossEntropyLoss::forwardMatrix(const Eigen::MatrixXf& predictions, const Eigen::MatrixXf& targets) {
    // Clip predictions to avoid log(0) for numerical stability
    this->predictions = predictions;
    this->targets = targets;
    Eigen::MatrixXf clipped_preds = predictions.cwiseMax(1e-7f).cwiseMin(1.0f - 1e-7f); // Clip predictions, epsilon = 1e-7
    // Cross-entropy loss: -1/n * sum(y_true * log(y_pred))
    return - (targets.array() * clipped_preds.array().log()).sum() / predictions.rows();  // Use predictions.rows() for consistency
}

Eigen::MatrixXf CategoricalCrossEntropyLoss::backwardMatrix() const {
    // Avoid divide by zero in log
    Eigen::MatrixXf safe_predictions = predictions.cwiseMax(1e-7f);  // Ensure clipping is applied to predictions

    // Compute gradient element-wise
    return -targets.cwiseQuotient(safe_predictions);
}

// Forward pass: Calculate loss for binary classification
float BinaryCrossEntropyLoss::forwardMatrix(const Eigen::MatrixXf& predictions, const Eigen::MatrixXf& targets) {
    this->predictions = predictions;
    this->targets = targets;

    // Clip predictions to avoid log(0) for numerical stability
    Eigen::MatrixXf clipped_preds = predictions.cwiseMax(1e-7f).cwiseMin(1.0f - 1e-7f);

    // CrossEntropyLoss = - sum(targets * log(predictions) + (1 - targets) * log(1 - predictions))
    return - (targets.array() * clipped_preds.array().log() + (1 - targets.array()) * (1 - clipped_preds.array()).log()).sum() / targets.rows();
}

Eigen::MatrixXf BinaryCrossEntropyLoss::backwardMatrix() const {
    // Avoid divide by zero in log
    Eigen::MatrixXf safe_predictions = predictions.cwiseMax(1e-7f);

    // Compute gradient for binary cross entropy
    return (safe_predictions - targets).cwiseQuotient(safe_predictions.cwiseProduct(Eigen::MatrixXf::Constant(safe_predictions.rows(), safe_predictions.cols(), 1.0f) - safe_predictions));
}

float CrossEntropyLoss::forwardMatrix(const Eigen::MatrixXf& logits, const Eigen::MatrixXf& targets) {
    // Compute softmax
    // Eigen::MatrixXf exp_logits = logits.array().exp();
    Eigen::MatrixXf exp_logits = (logits.array().colwise() - logits.rowwise().maxCoeff().array()).exp();
    Eigen::VectorXf row_sums = exp_logits.array().rowwise().sum();
    float epsilon = 1e-10f;

    this->predictions = exp_logits.array().colwise() / (row_sums.array() + epsilon);

    this->targets = targets;

    Eigen::MatrixXf clipped_preds = this->predictions.cwiseMax(1e-8f).cwiseMin(1.0f - 1e-8f);

    return - (targets.array() * clipped_preds.array().log()).sum() / logits.rows();
}

Eigen::MatrixXf CrossEntropyLoss::backwardMatrix() const {
    return predictions - targets;
}