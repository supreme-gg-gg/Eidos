#include "../include/optimizer.h"
#include "../include/layer.h"
#include <cmath>
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

// Function to initialize moment estimates for a layer
Adam::Moments Adam::initialize_moments(Layer& layer) {
    Moments moment;

    // Initialize moment estimates for weights
    Eigen::MatrixXf* grad_weights = layer.get_grad_weights();
    moment.m_w = Eigen::MatrixXf::Zero(grad_weights->rows(), grad_weights->cols());
    moment.v_w = Eigen::MatrixXf::Zero(grad_weights->rows(), grad_weights->cols());

    // Initialize moment estimates for bias
    if (layer.has_bias()) {
        Eigen::VectorXf* grad_bias = layer.get_grad_bias();
        moment.m_b = Eigen::VectorXf::Zero(grad_bias->rows());
        moment.v_b = Eigen::VectorXf::Zero(grad_bias->rows());
    }

    return moment;
}

// Adam optimizer's optimization step
void Adam::optimize(Layer& layer) {
    t++;  // Increment time step

    if (!layer.has_weights()) {
        return;
    }

    // Initialize moments if not already done
    if (moments.find(&layer) == moments.end()) {
        moments[&layer] = initialize_moments(layer);
    }

    // Access the moments for this layer
    Moments& moment = moments[&layer];

    // Update weights
    Eigen::MatrixXf* weights = layer.get_weights();
    Eigen::MatrixXf* grad_weights = layer.get_grad_weights();

    moment.m_w = beta1 * moment.m_w + (1 - beta1) * (*grad_weights);
    moment.v_w = beta2 * moment.v_w + (1 - beta2) * grad_weights->cwiseProduct(*grad_weights);

    Eigen::MatrixXf m_hat = moment.m_w / (1 - std::pow(beta1, t));
    Eigen::MatrixXf v_hat = moment.v_w / (1 - std::pow(beta2, t));

    *weights -= learning_rate * m_hat.cwiseQuotient(v_hat.cwiseSqrt() + Eigen::MatrixXf::Constant(v_hat.rows(), v_hat.cols(), epsilon));

    // Update bias if exists
    if (layer.has_bias()) {
        Eigen::VectorXf* bias = layer.get_bias();
        Eigen::VectorXf* grad_bias = layer.get_grad_bias();

        moment.m_b = beta1 * moment.m_b + (1 - beta1) * (*grad_bias);
        moment.v_b = beta2 * moment.v_b + (1 - beta2) * grad_bias->cwiseProduct(*grad_bias);

        Eigen::VectorXf m_hat_bias = moment.m_b / (1 - std::pow(beta1, t));
        Eigen::VectorXf v_hat_bias = moment.v_b / (1 - std::pow(beta2, t));

        *bias -= learning_rate * m_hat_bias.cwiseQuotient(v_hat_bias.cwiseSqrt() + Eigen::VectorXf::Constant(v_hat_bias.rows(), v_hat_bias.cols(), epsilon));
    }
}