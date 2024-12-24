#include "../include/layers/gru_layer.h"
#include <Eigen/Dense>
#include <iostream>

GRULayer::GRULayer(int input_size, int hidden_size, int output_size, Activation* activation, Activation* gate_activation, bool output_sequence) 
        : activation(activation), gate_activation(gate_activation), output_sequence(output_sequence){
    
    // Initialize weights and biases
    weights.push_back(Eigen::MatrixXf::Random(hidden_size, input_size));  // W_r: H x D
    weights.push_back(Eigen::MatrixXf::Random(hidden_size, hidden_size)); // U_r: H x H
    weights.push_back(Eigen::MatrixXf::Random(hidden_size, input_size));  // W_z: H x D
    weights.push_back(Eigen::MatrixXf::Random(hidden_size, hidden_size)); // U_z: H x H
    weights.push_back(Eigen::MatrixXf::Random(hidden_size, input_size));  // W_h: H x D
    weights.push_back(Eigen::MatrixXf::Random(hidden_size, hidden_size)); // U_h: H x H
    weights.push_back(Eigen::MatrixXf::Random(output_size, hidden_size));  // W_o: O x H
    
    biases.push_back(Eigen::VectorXf::Random(hidden_size));  // b_r: H
    biases.push_back(Eigen::VectorXf::Random(hidden_size));  // b_z: H
    biases.push_back(Eigen::VectorXf::Random(hidden_size));  // b_h: H
    biases.push_back(Eigen::VectorXf::Random(output_size));  // b_o: O

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
    hidden_state = Eigen::VectorXf::Zero(hidden_size);
}

Tensor GRULayer::forward(const Tensor& input) {
    this->input_sequence = input.getSingleMatrix();
    int T = input_sequence.rows();
    int D = input_sequence.cols();
    int H = hidden_state.rows();

    // Initialize hidden states to store all hidden states during the sequence
    hidden_states.resize(T + 1, Eigen::VectorXf::Zero(H));
    hidden_states[0] = hidden_state;  // Store the initial hidden state

    // Initialize gates and candidate states
    reset_gates.resize(T, Eigen::VectorXf::Zero(H));
    update_gates.resize(T, Eigen::VectorXf::Zero(H));
    candidate_states.resize(T, Eigen::VectorXf::Zero(H));

    // To store outputs at each time step
    Eigen::MatrixXf outputs = Eigen::MatrixXf::Zero(T, weights[6].rows());

    for (int i = 0; i < T; ++i) {
        Eigen::VectorXf x_t = input_sequence.row(i);

        // Reset gate: r_t = σ(W_r * x_t + U_r * h_{t-1} + b_r)
        Eigen::VectorXf r_t = activation->forward(weights[0] * x_t + weights[1] * hidden_states[i] + biases[0]);
        reset_gates[i] = r_t;

        // Update gate: z_t = σ(W_z * x_t + U_z * h_{t-1} + b_z)
        Eigen::VectorXf z_t = activation->forward(weights[2] * x_t + weights[3] * hidden_states[i] + biases[1]);
        update_gates[i] = z_t;

        // Candidate state: h_t_candidate = tanh(W_h * x_t + U_h * (r_t * h_{t-1}) + b_h)
        Eigen::VectorXf r_h_t_product = r_t.array() * hidden_states[i].array(); // Element-wise multiplication (r_t and hidden_states[i] are vectors)
        Eigen::VectorXf h_t_candidate = gate_activation->forward(weights[4] * x_t + weights[5] * r_h_t_product + biases[2]);

        // Hidden state: h_t = (1 - z_t) * h_{t-1} + z_t * h_t_candidate
        hidden_states[i + 1] = (1 - z_t.array()) * hidden_states[i].array() + z_t.array() * h_t_candidate.array();

        // Output: o_t = W_o * h_t + b_o
        if (output_sequence) {
            outputs.row(i) = weights[6] * hidden_states[i+1] + biases[3];
        }
    }

    // Update the hidden_state member to the final state for continuity
    this->hidden_state = hidden_states[T];

    if (output_sequence) {
        return Tensor(outputs);
    } else {
        return Tensor(hidden_states[T].transpose());
    }
}

Tensor GRULayer::backward(const Tensor& grad_output_sequence) {
    int T = input_sequence.rows();
    int D = input_sequence.cols();
    int H = hidden_state.rows();

    Eigen::MatrixXf grad_output_sequence_mat = grad_output_sequence.getSingleMatrix();

    // Zero out gradients for weights and biases
    for (int i = 0; i < 7; ++i) {
        grad_weights[i].setZero();
    }
    for (int i = 0; i < 4; ++i) {
        grad_biases[i].setZero();
    }

    // Initialize gradient of next hidden state
    Eigen::VectorXf grad_h_next = Eigen::VectorXf::Zero(H);

    // Gradient of final output (if output_sequence is false)
    // This is the gradient of the loss with respect to the final hidden state
    if (!output_sequence) {
        grad_h_next = grad_output_sequence_mat.transpose();
    }

    // Iterate backwards through the sequence
    for (int t = T - 1; t >= 0; --t) {
        Eigen::VectorXf x_t = input_sequence.row(t);
        Eigen::VectorXf r_t = reset_gates[t];
        Eigen::VectorXf z_t = update_gates[t];
        Eigen::VectorXf h_t_candidate = candidate_states[t];
        Eigen::VectorXf h_prev = (t > 0) ? hidden_states[t] : hidden_state;

        Eigen::VectorXf grad_h_t = grad_h_next; // take hidden state grad from the last time step
        grad_h_t += weights[6].transpose() * grad_output_sequence_mat.row(t).transpose(); // add gradient from output layer

        // reset gate
        Eigen::VectorXf grad_r_t = grad_h_t.array() * (h_t_candidate.array() - h_prev.array()) * (activation->backward(r_t)).array();
        // reset gate weights
        grad_weights[0] += grad_r_t * x_t.transpose();
        grad_weights[1] += grad_r_t * h_prev.transpose();
        grad_biases[0] += grad_r_t;

        // update gate
        Eigen::VectorXf grad_z_t = grad_h_t.array() * (h_t_candidate.array() - h_prev.array()) * (activation->backward(z_t)).array();
        // update gate weights
        grad_weights[2] += grad_z_t * x_t.transpose();
        grad_weights[3] += grad_z_t * h_prev.transpose();
        grad_biases[1] += grad_z_t;

        // candidate state
        Eigen::VectorXf grad_h_candidate = grad_h_t.array() * z_t.array() * (gate_activation->backward(h_t_candidate)).array();
        // candidate state weights
        grad_weights[4] += grad_h_candidate * x_t.transpose();
        grad_weights[5] += grad_h_candidate * (r_t.array() * h_prev.array()).matrix().transpose();
        grad_biases[2] += grad_h_candidate;

        // previous hidden state
        grad_h_next = (grad_h_t.array() * (1 - z_t.array())).matrix()
            + weights[1].transpose() * (grad_r_t.array() * (activation->backward(r_t)).array()).matrix() 
            + weights[3].transpose() * (grad_h_candidate.array() * r_t.array() * (gate_activation->backward(grad_h_candidate)).array()).matrix();
    }

    // Gradient with respect to initial hidden state
    return Tensor(grad_h_next);
}

void GRULayer::serialize(std::ofstream& toFileStream) const {
    // Serialize weights
    for (const auto& weight : weights) {
        Eigen::Index rows = weight.rows();
        Eigen::Index cols = weight.cols();
        toFileStream.write((char*)&rows, sizeof(Eigen::Index));
        toFileStream.write((char*)&cols, sizeof(Eigen::Index));
        toFileStream.write((char*)weight.data(), rows * cols * sizeof(float));
    }

    // Serialize biases
    for (const auto& bias : biases) {
        Eigen::Index rows = bias.rows();
        toFileStream.write((char*)&rows, sizeof(Eigen::Index));
        toFileStream.write((char*)bias.data(), rows * sizeof(float));
    }
}

GRULayer* GRULayer::deserialize(std::ifstream& fromFileStream) {
    std::vector<Eigen::MatrixXf> weights;
    std::vector<Eigen::VectorXf> biases;

    for (int i = 0; i < 7; ++i) {
        Eigen::Index rows, cols;
        fromFileStream.read((char*)&rows, sizeof(Eigen::Index));
        fromFileStream.read((char*)&cols, sizeof(Eigen::Index));
        Eigen::MatrixXf weight(rows, cols);
        fromFileStream.read((char*)weight.data(), rows * cols * sizeof(float));
        weights.push_back(weight);
    }

    for (int i = 0; i < 4; ++i) {
        Eigen::Index rows;
        fromFileStream.read((char*)&rows, sizeof(Eigen::Index));
        Eigen::VectorXf bias(rows);
        fromFileStream.read((char*)bias.data(), rows * sizeof(float));
        biases.push_back(bias);
    }

    GRULayer* layer = new GRULayer(weights[0].cols(), weights[0].rows(), weights[6].rows(), nullptr, nullptr, false);
    layer->weights = weights;
    layer->biases = biases;
    return layer;
}