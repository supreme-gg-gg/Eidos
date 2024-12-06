#include "../include/optimizer.h"
#include <Eigen/Dense>

void SGD::optimize(Layer& layer) {
    if (layer.has_weights()) {
        Eigen::MatrixXf* weights = layer.get_weights();
        Eigen::MatrixXf* grad_weights = layer.get_grad_weights();
        *weights -= learning_rate * (*grad_weights);
    }
    if (layer.has_bias()) {
        Eigen::VectorXf* bias = layer.get_bias();
        Eigen::VectorXf* grad_bias = layer.get_grad_bias();
        *bias -= learning_rate * (*grad_bias);
    }
}

void SGD::update(Eigen::MatrixXf& weights, const Eigen::MatrixXf& weight_gradients,
        Eigen::VectorXf* bias = nullptr, const Eigen::VectorXf* bias_gradients = nullptr) {
    // Update weights
    weights -= learning_rate * weight_gradients;

    // Update bias, if it exists
    if (bias != nullptr && bias_gradients != nullptr) {
        *bias -= learning_rate * (*bias_gradients);
    }
}

/*
void Adam::update(Eigen::MatrixXf& parameters, const Eigen::MatrixXf& gradients, Eigen::VectorXf* bias = nullptr, const Eigen::VectorXf* bias_gradients = nullptr) {
    t++; // Increment time step

    Eigen::MatrixXf m = Eigen::MatrixXf::Zero(gradients.rows(), gradients.cols());
    Eigen::MatrixXf v = Eigen::MatrixXf::Zero(gradients.rows(), gradients.cols());
    Eigen::VectorXf m_bias = Eigen::VectorXf::Zero(bias_gradients->rows());
    Eigen::VectorXf v_bias = Eigen::VectorXf::Zero(bias_gradients->rows());

    // Update moment estimates for parameters
    m = beta1 * m + (1 - beta1) * gradients; // mt = beta1 * mt-1 + (1 - beta1) * gt
    v = beta2 * v + (1 - beta2) * gradients.cwiseProduct(gradients); // vt = beta2 * vt-1 + (1 - beta2) * gt^2

    // Bias correction of moments
    Eigen::MatrixXf m_hat = m / (1 - std::pow(beta1, t)); // m_hat = mt / (1 - beta1^t)
    Eigen::MatrixXf v_hat = v / (1 - std::pow(beta2, t)); // v_hat = vt / (1 - beta2^t)

    // Update weights: w = w - alpha * m_hat / (sqrt(v_hat) + epsilon)
    parameters -= learning_rate * m_hat.cwiseQuotient(v_hat.cwiseSqrt() + Eigen::MatrixXf::Constant(v.rows(), v.cols(), epsilon));
    
    // If there's a bias, handle its update separately
    if (bias != nullptr && bias_gradients != nullptr) {
        m_bias = beta1 * m_bias + (1 - beta1) * (*bias_gradients);
        v_bias = beta2 * v_bias + (1 - beta2) * (*bias_gradients).cwiseProduct(*bias_gradients);

        // Bias correction
        Eigen::VectorXf m_hat_bias = m_bias / (1 - std::pow(beta1, t));
        Eigen::VectorXf v_hat_bias = v_bias / (1 - std::pow(beta2, t));

        // Update bias: b = b - alpha * m_hat_bias / (sqrt(v_hat_bias) + epsilon)
        *bias -= learning_rate * m_hat_bias.cwiseQuotient(v_hat_bias.cwiseSqrt() + Eigen::VectorXf::Constant(v_bias.rows(), v_bias.cols(), epsilon));
    }
}

void Adam::optimize(Layer& layer) {
    t++;  // Increment time step

    if (!layer.has_weights()) {
        return;
    }

    // Initialize moment and velocity matrices for weights
    Eigen::MatrixXf& parameters = layer.get_weights(); // Assuming get_weights() exists
    Eigen::MatrixXf gradients = layer.get_gradients(); // Assuming get_gradients() exists

    Eigen::MatrixXf m = Eigen::MatrixXf::Zero(gradients.rows(), gradients.cols());
    Eigen::MatrixXf v = Eigen::MatrixXf::Zero(gradients.rows(), gradients.cols());
    
    // For bias handling, assuming you have get_biases() and get_bias_gradients() methods in Layer
    Eigen::VectorXf m_bias = Eigen::VectorXf::Zero(layer.get_bias_gradients().rows());
    Eigen::VectorXf v_bias = Eigen::VectorXf::Zero(layer.get_bias_gradients().rows());

    // Update moment estimates for weights
    m = beta1 * m + (1 - beta1) * gradients;  // m = beta1 * m + (1 - beta1) * grad
    v = beta2 * v + (1 - beta2) * gradients.cwiseProduct(gradients);  // v = beta2 * v + (1 - beta2) * grad^2

    // Bias correction
    Eigen::MatrixXf m_hat = m / (1 - std::pow(beta1, t));  // m_hat = m / (1 - beta1^t)
    Eigen::MatrixXf v_hat = v / (1 - std::pow(beta2, t));  // v_hat = v / (1 - beta2^t)

    // Update weights
    parameters -= learning_rate * m_hat.cwiseQuotient(v_hat.cwiseSqrt() + Eigen::MatrixXf::Constant(v.rows(), v.cols(), epsilon));

    // Handle bias update
    if (layer.has_bias()) {
        Eigen::VectorXf bias = layer.get_bias();
        Eigen::VectorXf bias_gradients = layer.get_bias_gradients();

        m_bias = beta1 * m_bias + (1 - beta1) * bias_gradients;
        v_bias = beta2 * v_bias + (1 - beta2) * bias_gradients.cwiseProduct(bias_gradients);

        // Bias correction
        Eigen::VectorXf m_hat_bias = m_bias / (1 - std::pow(beta1, t));
        Eigen::VectorXf v_hat_bias = v_bias / (1 - std::pow(beta2, t));

        // Update bias
        bias -= learning_rate * m_hat_bias.cwiseQuotient(v_hat_bias.cwiseSqrt() + Eigen::VectorXf::Constant(v_bias.rows(), v_bias.cols(), epsilon));
    }
}
*/