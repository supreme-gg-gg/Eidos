#include "../include/cross_entropy_loss.h"
#include <Eigen/Dense>

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