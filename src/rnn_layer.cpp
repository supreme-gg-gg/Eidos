#include <Eigen/Dense>
#include "../include/rnn_layer.h"

RNNLayer::RNNLayer(int input_size, int hidden_size, Activation* activation, bool output_sequence) 
    : activation(activation), output_sequence(output_sequence) {
    W_h = Eigen::MatrixXf::Random(hidden_size, input_size);
    U_h = Eigen::MatrixXf::Random(hidden_size, hidden_size);
    b_h = Eigen::VectorXf::Random(hidden_size);

    if (output_sequence) {
        W_o = Eigen::MatrixXf::Random(hidden_size, hidden_size);
        b_o = Eigen::VectorXf::Random(hidden_size);
    } 
}

Eigen::MatrixXf RNNLayer::forward(const Eigen::MatrixXf& input) {
    int seq_len = input.rows();
    int input_size = input.cols();
    int hidden_size = W_h.rows();
    hidden_state = Eigen::MatrixXf::Zero(seq_len, hidden_size);

    for (int t = 0; t < seq_len; ++t) {
        Eigen::VectorXf x_t = input.row(t);
        Eigen::VectorXf h_t = hidden_state.row(t);

        Eigen::VectorXf h_t_next = W_h * x_t + U_h * h_t + b_h;
        h_t_next = activation->forward(h_t_next);
        hidden_state.row(t + 1) = h_t_next;
    }

    return W_o * hidden_state.transpose() + b_o.replicate(1, seq_len);
}

Eigen::MatrixXf RNNLayer::backward(const Eigen::MatrixXf& grad_output) {
    int seq_len = grad_output.cols();
    int hidden_size = W_h.rows();
    Eigen::MatrixXf grad_hidden = W_o.transpose() * grad_output;

    Eigen::MatrixXf grad_W_h = Eigen::MatrixXf::Zero(W_h.rows(), W_h.cols());
    Eigen::MatrixXf grad_U_h = Eigen::MatrixXf::Zero(U_h.rows(), U_h.cols());
    Eigen::VectorXf grad_b_h = Eigen::VectorXf::Zero(b_h.size());
    Eigen::MatrixXf grad_W_o = Eigen::MatrixXf::Zero(W_o.rows(), W_o.cols());
    Eigen::VectorXf grad_b_o = Eigen::VectorXf::Zero(b_o.size());

    Eigen::MatrixXf grad_h_next = Eigen::VectorXf::Zero(hidden_size);
    for (int t = seq_len - 1; t >= 0; --t) {
        Eigen::VectorXf x_t = input.row(t);
        Eigen::VectorXf h_t = hidden_state.row(t);
        Eigen::VectorXf h_t_next = hidden_state.row(t + 1);

        Eigen::VectorXf grad_h = grad_hidden.col(t) + grad_h_next;
        Eigen::VectorXf grad_h_t = activation->backward(grad_h);

        grad_W_h += grad_h_t * x_t.transpose();
        grad_U_h += grad_h_t * h_t.transpose();
        grad_b_h += grad_h_t;

        grad_h_next = U_h.transpose() * grad_h_t;
    }

    grad_W_o = grad_output * hidden_state.transpose();
    grad_b_o = grad_output.rowwise().sum();

    return grad_hidden;
}

bool RNNLayer::has_weights() const {
    return true;
}

bool RNNLayer::has_bias() const {
    return true;
}

Eigen::MatrixXf* RNNLayer::get_weights() {
    return &W_h;
}

Eigen::MatrixXf* RNNLayer::get_grad_weights() {
    return &grad_W_h;
}

Eigen::VectorXf* RNNLayer::get_bias() {
    return &b_h;
}

Eigen::VectorXf* RNNLayer::get_grad_bias() {
    return &grad_b_h;
}