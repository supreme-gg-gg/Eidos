#include <Eigen/Dense>
#include "../include/rnn_layer.h"

// Constructor
RNNLayer::RNNLayer(int input_size, int hidden_size, Activation* activation, bool output_sequence)
    : activation(activation), output_sequence(output_sequence) {
    // Initialize weights and biases
    weights.push_back(Eigen::MatrixXf::Random(hidden_size, input_size));  // W_h
    weights.push_back(Eigen::MatrixXf::Random(hidden_size, hidden_size)); // U_h
    weights.push_back(Eigen::MatrixXf::Random(hidden_size, hidden_size)); // W_o

    biases.push_back(Eigen::VectorXf::Random(hidden_size));  // b_h
    biases.push_back(Eigen::VectorXf::Random(hidden_size));  // b_o

    // Initialize gradients
    grad_weights.resize(weights.size());
    grad_biases.resize(biases.size());
    for (size_t i = 0; i < weights.size(); ++i) {
        grad_weights[i] = Eigen::MatrixXf::Zero(weights[i].rows(), weights[i].cols());
    }
    for (size_t i = 0; i < biases.size(); ++i) {
        grad_biases[i] = Eigen::VectorXf::Zero(biases[i].rows());
    }

    // Initialize hidden state
    hidden_state = Eigen::MatrixXf::Zero(hidden_size, 1);
}

Eigen::MatrixXf RNNLayer::forward(const Eigen::MatrixXf& input_sequence) {
    int T = input_sequence.rows();  // Sequence length
    int D = input_sequence.cols();  // Input size
    int H = hidden_state.rows();    // Hidden size

    this->input_sequence = input_sequence;  // Store input sequence for backpropagation

    std::vector<Eigen::VectorXf> outputs;  // To store outputs at each time step

    for (int t = 0; t < T; ++t) {
        // Input at time step t
        Eigen::VectorXf x_t = input_sequence.row(t);

        // hidden_state = activation(W_h * x_t + U_h * hidden_state + b_h)
        hidden_state = activation->forward(weights[0] * x_t + weights[1] * hidden_state.col(0) + biases[0]);

        // Compute output for the current time step if output_sequence is true
        if (output_sequence) {
            Eigen::VectorXf o_t = weights[2] * hidden_state + biases[1];  // W_o * h_t + b_o
            outputs.push_back(o_t);
        }
    }

    if (output_sequence) {
        // Convert vector of outputs to a matrix (T x output_size)
        Eigen::MatrixXf all_outputs(T, outputs[0].rows());
        for (int t = 0; t < T; ++t) {
            all_outputs.row(t) = outputs[t].transpose();
        }
        return all_outputs;  // Return all outputs as a matrix
    } else {
        // If output_sequence is false, return the last hidden state
        return hidden_state;
    }
}

// Backward pass
Eigen::MatrixXf RNNLayer::backward(const Eigen::MatrixXf& grad_output_sequence) {
    int T = grad_output_sequence.rows();  // Sequence length
    int H = hidden_state.rows();          // Hidden size

    grad_weights[0].setZero();  // W_h
    grad_weights[1].setZero();  // U_h
    grad_weights[2].setZero();  // W_o
    grad_biases[0].setZero();   // b_h
    grad_biases[1].setZero();   // b_o

    Eigen::MatrixXf grad_h_next = Eigen::MatrixXf::Zero(H, 1);  // Gradient of h_{t+1}

    for (int t = T - 1; t >= 0; --t) {
        // Gradient w.r.t. output: dL/dy_t = dL/dy_t + dL/dh_t * dh_t/dy_t
        Eigen::VectorXf grad_o_t = grad_output_sequence.row(t);
        grad_weights[2] += grad_o_t * hidden_state(t);
        grad_biases[1] += grad_o_t;

        // Backpropagate to hidden state t: dL/dh_t = W_o^T * dL/dy_t + dL/dh_{t+1} * dh_{t+1}/dh_t
        Eigen::VectorXf grad_h_t = weights[2].transpose() * grad_o_t + grad_h_next;

        // Backprop through activation function: dL/dh_t * dh_t/dz_t
        // @bug this should be wrong because the activation function is not applied to the hidden state
        Eigen::MatrixXf grad_h_t_raw = activation->backward(hidden_state) * grad_h_t;

        // Gradients for W_h, U_h, b_h
        grad_weights[0] += grad_h_t_raw * input_sequence(t); // dL/dW_h = dL/dh_t * dh_t/dW_h
        grad_weights[1] += grad_h_t_raw * hidden_state(t - 1); // dL/dU_h = dL/dh_t * dh_t/dU_h
        grad_biases[0] += grad_h_t_raw; // dL/db_h = dL/dh_t * dh_t/db_h

        // Propagate to the previous time step
        grad_h_next = weights[1].transpose() * grad_h_t_raw; // dL/dh_{t+1} = U_h^T * dL/dh_t
    }

    return grad_h_next;  // Return gradient w.r.t. input (optional)
}

// Optimizer interface methods
bool RNNLayer::has_weights() const {
    return true;
}

bool RNNLayer::has_bias() const {
    return true;
}

std::vector<Eigen::MatrixXf*> RNNLayer::get_weights() {
    std::vector<Eigen::MatrixXf*> pointers;
    for (auto& weight : weights) {
        pointers.push_back(&weight);
    }
    return pointers;
}

std::vector<Eigen::MatrixXf*> RNNLayer::get_grad_weights() {
    std::vector<Eigen::MatrixXf*> pointers;
    for (auto& grad_weight : grad_weights) {
        pointers.push_back(&grad_weight);
    }
    return pointers;
}

std::vector<Eigen::VectorXf*> RNNLayer::get_bias() {
    std::vector<Eigen::VectorXf*> pointers;
    for (auto& bias : biases) {
        pointers.push_back(&bias);
    }
    return pointers;
}

std::vector<Eigen::VectorXf*> RNNLayer::get_grad_bias() {
    std::vector<Eigen::VectorXf*> pointers;
    for (auto& grad_bias : grad_biases) {
        pointers.push_back(&grad_bias);
    }
    return pointers;
}