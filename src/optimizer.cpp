#include "../include/optimizer.h"
#include "../include/layer.h"
#include <cmath>
#include <Eigen/Dense>

void SGD::optimize(Layer& layer) {
    // Use const reference to bind to the returned vectors
    const auto& weights = layer.get_weights();
    const auto& grad_weights = layer.get_grad_weights();

    // Iterate over weights and gradients together
    for (size_t i = 0; i < weights.size(); ++i) {
        *weights[i] -= learning_rate * (*grad_weights[i]);  // Update weight using corresponding gradient
    }

    // Similar logic for bias if applicable
    if (layer.has_bias()) {
        const auto& bias = layer.get_bias();
        const auto& grad_bias = layer.get_grad_bias();

        for (size_t i = 0; i < bias.size(); ++i) {
            *bias[i] -= learning_rate * (*grad_bias[i]);  // Update bias using corresponding gradient
        }
    }
}

Adam::Moments Adam::initialize_moments(Layer& layer) {
    Moments moment;

    // Initialize moments for all weights
    const auto& grad_weights_list = layer.get_grad_weights();
    for (const auto& grad_weights : grad_weights_list) {
        moment.m_w.push_back(Eigen::MatrixXf::Zero(grad_weights->rows(), grad_weights->cols()));
        moment.v_w.push_back(Eigen::MatrixXf::Zero(grad_weights->rows(), grad_weights->cols()));
    }

    // Initialize moments for all biases
    const auto& grad_biases_list = layer.get_grad_bias();
    for (const auto& grad_bias : grad_biases_list) {
        moment.m_b.push_back(Eigen::VectorXf::Zero(grad_bias->rows()));
        moment.v_b.push_back(Eigen::VectorXf::Zero(grad_bias->rows()));
    }

    return moment;
}

// Adam optimizer's optimization step
void Adam::optimize(Layer& layer) {
    t++;  // Increment time step

    // Skip layers without weights
    if (layer.get_weights().empty()) {
        return;
    }

    // Initialize moments if not already done
    if (moments.find(&layer) == moments.end()) {
        moments[&layer] = initialize_moments(layer);
    }

    // Access the moments for this layer
    Moments& moment = moments[&layer];

    // Update weights
    const auto& weights_list = layer.get_weights();
    const auto& grad_weights_list = layer.get_grad_weights();
    for (size_t i = 0; i < weights_list.size(); ++i) {
        Eigen::MatrixXf* weights = weights_list[i];
        const Eigen::MatrixXf* grad_weights = grad_weights_list[i];

        moment.m_w[i] = beta1 * moment.m_w[i] + (1 - beta1) * (*grad_weights);
        moment.v_w[i] = beta2 * moment.v_w[i] + (1 - beta2) * grad_weights->cwiseProduct(*grad_weights);

        Eigen::MatrixXf m_hat = moment.m_w[i] / (1 - std::pow(beta1, t));
        Eigen::MatrixXf v_hat = moment.v_w[i] / (1 - std::pow(beta2, t));

        *weights -= learning_rate * m_hat.cwiseQuotient(v_hat.cwiseSqrt() + Eigen::MatrixXf::Constant(v_hat.rows(), v_hat.cols(), epsilon));
    }

    // Update biases
    const auto& biases_list = layer.get_bias();
    const auto& grad_biases_list = layer.get_grad_bias();
    for (size_t i = 0; i < biases_list.size(); ++i) {
        Eigen::VectorXf* bias = biases_list[i];
        const Eigen::VectorXf* grad_bias = grad_biases_list[i];

        moment.m_b[i] = beta1 * moment.m_b[i] + (1 - beta1) * (*grad_bias);
        moment.v_b[i] = beta2 * moment.v_b[i] + (1 - beta2) * grad_bias->cwiseProduct(*grad_bias);

        Eigen::VectorXf m_hat_bias = moment.m_b[i] / (1 - std::pow(beta1, t));
        Eigen::VectorXf v_hat_bias = moment.v_b[i] / (1 - std::pow(beta2, t));

        *bias -= learning_rate * m_hat_bias.cwiseQuotient(v_hat_bias.cwiseSqrt() + Eigen::VectorXf::Constant(v_hat_bias.rows(), epsilon));
    }
}