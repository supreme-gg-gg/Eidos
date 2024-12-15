#ifndef RNN_LAYER_H
#define RNN_LAYER_H

#include <Eigen/Dense>
#include "../layer.h"
#include "../activation_fns.h"

class RNNLayer : public Layer {
private:
    // Store weights and biases in vectors for flexibility
    std::vector<Eigen::MatrixXf> weights;        // Weight matrices (W_h, U_h, W_o)
    std::vector<Eigen::VectorXf> biases;         // Bias vectors (b_h, b_o)
    std::vector<Eigen::MatrixXf> grad_weights;   // Gradients of weights
    std::vector<Eigen::VectorXf> grad_biases;    // Gradients of biases

    Eigen::VectorXf hidden_state;  // Current hidden state
    std::vector<Eigen::VectorXf> hidden_states;  // Hidden states for each time step
    
    Activation* activation;        // Activation function
    std::vector<Eigen::VectorXf> pre_activations;

    bool output_sequence;
    Eigen::MatrixXf input_sequence;

public:
    RNNLayer(int input_size, int hidden_size, int output_size, Activation* activation, bool output_sequence = false);

    // Forward pass through the RNN layer
    Eigen::MatrixXf forward(const Eigen::MatrixXf& input) override;

    // Backward pass for the RNN layer
    Eigen::MatrixXf backward(const Eigen::MatrixXf& grad_output) override;

    bool has_weights() const { return true; }
    bool has_bias() const { return true; }

    std::vector<Eigen::MatrixXf*> get_weights() { return get_pointers(weights); }
    std::vector<Eigen::MatrixXf*> get_grad_weights() { return get_pointers(grad_weights); }
    std::vector<Eigen::VectorXf*> get_bias() { return get_pointers(biases); }
    std::vector<Eigen::VectorXf*> get_grad_bias() { return get_pointers(grad_biases); }
    
    ~RNNLayer() = default;

protected:
    template <typename T>
    std::vector<T*> get_pointers(std::vector<T>& vec) {
        std::vector<T*> pointers;
        for (auto& item : vec) {
            pointers.push_back(&item);
        }
        return pointers;
    }
};

#endif //RNN_LAYER_H