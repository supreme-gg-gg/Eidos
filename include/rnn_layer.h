#ifndef RNN_LAYER_H
#define RNN_LAYER_H

#include <Eigen/Dense>
#include "layer.h"
#include "activation_fns.h"

class RNNLayer : public Layer {
private:
    Eigen::MatrixXf W_h, U_h, W_o;  // Weight matrices
    Eigen::VectorXf b_h, b_o;      // Bias vectors
    Eigen::MatrixXf hidden_state;  // Current hidden state
    Eigen::MatrixXf grad_W_h, grad_U_h, grad_W_o;  // Gradient of weights
    Eigen::VectorXf grad_b_h, grad_b_o;            // Gradient of biases
    Activation* activation;        // Activation function

    bool output_sequence;

public:
    RNNLayer(int input_size, int hidden_size, Activation* activation, bool output_sequence = false);

    // Forward pass through the RNN layer
    Eigen::MatrixXf forward(const Eigen::MatrixXf& input) override;

    // Backward pass for the RNN layer
    Eigen::MatrixXf backward(const Eigen::MatrixXf& grad_output) override;

    bool has_weights() const override;
    bool has_bias() const override;

    Eigen::MatrixXf* get_weights() override;
    Eigen::MatrixXf* get_grad_weights() override;

    Eigen::VectorXf* get_bias() override;
    Eigen::VectorXf* get_grad_bias() override;

    ~RNNLayer() = default;
};