#include <Eigen/Dense>
#include <iostream>
#include "../include/layers/rnn_layer.h"

// Constructor
RNNLayer::RNNLayer(int input_size, int hidden_size, int output_size, Activation* activation, bool output_sequence)
    : activation(activation), output_sequence(output_sequence) {
    // Initialize weights and biases
    weights.push_back(Eigen::MatrixXf::Random(hidden_size, input_size));  // W_h: H x D
    weights.push_back(Eigen::MatrixXf::Random(hidden_size, hidden_size)); // U_r: H x H
    weights.push_back(Eigen::MatrixXf::Random(output_size, hidden_size)); // W_o: O x H

    biases.push_back(Eigen::VectorXf::Random(hidden_size));  // b_h: H
    biases.push_back(Eigen::VectorXf::Random(output_size));  // b_o: O

    // Initialize gradients
    grad_weights.resize(weights[0].size());
    grad_biases.resize(biases[0].size());
    for (size_t i = 0; i < weights.size(); ++i) {
        grad_weights[i] = Eigen::MatrixXf::Zero(weights[i].rows(), weights[i].cols());
    }
    for (size_t i = 0; i < biases.size(); ++i) {
        grad_biases[i] = Eigen::VectorXf::Zero(biases[i].rows());
    }

    // Initialize hidden state
    hidden_state = Eigen::VectorXf::Zero(hidden_size);
}

Tensor RNNLayer::forward(const Tensor& input_sequence) {

    this->input_sequence = input_sequence.getSingleMatrix();  // Store input sequence for backpropagation

    int T = this->input_sequence.rows();  // Sequence length
    int D = this->input_sequence.cols();  // Input size
    int H = hidden_state.rows();    // Hidden size


    // Pre activations for gradient computation
    this->pre_activations.resize(T, Eigen::VectorXf::Zero(H));
    
    // Initialize hidden_states to store all hidden states during the sequence
    this->hidden_states.resize(T + 1, Eigen::VectorXf::Zero(H));
    hidden_states[0] = hidden_state;  // Store the initial hidden state

    // To store outputs at each time step
    Eigen::MatrixXf outputs = Eigen::MatrixXf::Zero(T, weights[2].rows());  

    for (int t = 0; t < T; ++t) {
        Eigen::VectorXf x_t = this->input_sequence.row(t);
        
        // Compute pre-activation and store it
        Eigen::VectorXf pre_activation = weights[0] * x_t + weights[1] * hidden_states[t] + biases[0];
        pre_activations[t] = pre_activation;

        // Apply activation function to get the next hidden state
        hidden_states[t + 1] = activation->forward(pre_activation);

        // If output sequence is required, calculate and store the output
        if (output_sequence) {
            Eigen::VectorXf o_t = weights[2] * hidden_states[t + 1] + biases[1];
            outputs.row(t) = o_t.transpose();
        }
    }

    // Update the hidden_state member to the final state for continuity
    hidden_state = hidden_states[T];

    if (output_sequence) {
        return Tensor(outputs); // Return output sequence
    } else {
        // If output_sequence is false, return the last hidden state
        return Tensor(hidden_states[T]);
    }
}

// Backward pass
Tensor RNNLayer::backward(const Tensor& grad_output_sequence) {

    Eigen::MatrixXf grad_output_mat = grad_output_sequence.getSingleMatrix();

    int T = grad_output_mat.rows();  // Sequence length
    int H = hidden_state.rows();          // Hidden size

    grad_weights[0].setZero();  // W_h
    grad_weights[1].setZero();  // U_h
    grad_weights[2].setZero();  // W_o
    grad_biases[0].setZero();   // b_h
    grad_biases[1].setZero();   // b_o

    Eigen::MatrixXf grad_h_next = Eigen::MatrixXf::Zero(H, 1);  // Gradient of h_{t+1}

    for (int t = T - 1; t >= 0; --t) {
        Eigen::VectorXf grad_o_t = grad_output_mat.row(t).transpose();

        if (output_sequence) {
            grad_weights[2] += grad_o_t * hidden_states[t + 1].transpose();
            grad_biases[1] += grad_o_t;
        }

        Eigen::VectorXf grad_h_t = weights[2].transpose() * grad_o_t + grad_h_next;
        Eigen::VectorXf grad_h_t_raw = (activation->backward(pre_activations[t])).array() * grad_h_t.array();

        grad_weights[0] += grad_h_t_raw * input_sequence.row(t);
        grad_weights[1] += grad_h_t_raw * hidden_states[t].transpose();
        grad_biases[0] += grad_h_t_raw;

        grad_h_next = weights[1].transpose() * grad_h_t_raw;
    }

    return Tensor(grad_h_next);  // Return gradient w.r.t. input (optional)
}