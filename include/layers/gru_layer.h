#ifndef GRU_LAYER_H
#define GRU_LAYER_H

#include "rnn_layer.h"
#include "../activations.h"
#include <vector>
#include <Eigen/Dense>

class GRULayer : public RNNLayer {
private:
    // Store weights and biases in vectors for flexibility
    std::vector<Eigen::MatrixXf> weights;        // Weight matrices (W_r, U_r, W_z, U_z, W_h, U_h, W_o)
    std::vector<Eigen::VectorXf> biases;         // Bias vectors (b_r, b_z, b_h, b_o)
    std::vector<Eigen::MatrixXf> grad_weights;   // Gradients of weights
    std::vector<Eigen::VectorXf> grad_biases;    // Gradients of biases

    Eigen::VectorXf hidden_state;  // Current hidden state
    std::vector<Eigen::VectorXf> hidden_states;  // Hidden states for each time step

    std::vector<Eigen::VectorXf> reset_gates;    // Reset gate activations
    std::vector<Eigen::VectorXf> update_gates;   // Update gate activations
    std::vector<Eigen::VectorXf> candidate_states; // Candidate states

    Activation* activation;        // Activation function for candidate state
    Activation* gate_activation;   // Activation for gates (usually sigmoid)

    Eigen::MatrixXf input_sequence; // Input data stored for backward pass
    bool output_sequence;           // Whether to output the full sequence or just the last state

public:
    // Constructor
    GRULayer(int input_size, int hidden_size, int output_size, Activation* activation, Activation* gate_activation, bool output_sequence = false);

    // Forward pass through the GRU layer
    Eigen::MatrixXf forward(const Eigen::MatrixXf& input) override;

    // Backward pass for the GRU layer
    Eigen::MatrixXf backward(const Eigen::MatrixXf& grad_output) override;

    ~GRULayer() = default;
};

#endif // GRU_LAYER_HPP