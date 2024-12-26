#ifndef GRU_LAYER_H
#define GRU_LAYER_H

#include "../layer.h"
#include "../activations.h"
#include <vector>
#include <Eigen/Dense>

/**
 * @class GRULayer
 * @brief A Gated Recurrent Unit (GRU) layer for recurrent neural networks.
 * 
 * This class implements a GRU layer, which is a type of recurrent neural network (RNN) layer.
 * It includes methods for the forward and backward passes, as well as storing weights, biases,
 * and hidden states.
 * 
 * @details
 * The GRU layer consists of reset gates, update gates, and candidate states. It uses two 
 * activation functions: one for the gates (typically sigmoid) and one for the candidate state 
 * (typically tanh). The layer can output either the full sequence of hidden states or just the 
 * last hidden state, depending on the `output_sequence` parameter.
 * 
 * @param input_size The number of input features for the GRU layer.
 * @param hidden_size The number of hidden units in the GRU layer.
 * @param output_size The number of output features from the GRU layer.
 * @param activation The activation function for the reset and update gates.
 * @param gate_activation The activation function for the candidate state.
 * @param output_sequence Whether to output the full sequence of hidden states or just the last state.
 * 
 * @section Example
 * @code
 * Activation* sigmoid = new SigmoidActivation();
 * Activation* tanh = new TanhActivation();
 * GRULayer gru_layer(10, 20, 30, sigmoid, tanh, true);
 * Eigen::MatrixXf input = Eigen::MatrixXf::Random(5, 10); // Example input
 * Eigen::MatrixXf output = gru_layer.forward(input);
 * @endcode
 */
class GRULayer : public Layer {
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

    Activation* activation;        // Activation function for reset and update gates (sigmoid) 
    Activation* gate_activation;   // Activation for candidate state (tanh)

    Eigen::MatrixXf input_sequence; // Input data stored for backward pass
    bool output_sequence;           // Whether to output the full sequence or just the last state

public:
    // Constructor
    /**
     * @param input_size The number of input features for the GRU layer.
     */
    GRULayer(int input_size, int hidden_size, int output_size, Activation* activation, Activation* gate_activation, bool output_sequence = true);

    // Forward pass through the GRU layer
    Tensor forward(const Tensor& input) override;

    // Backward pass for the GRU layer
    Tensor backward(const Tensor& grad_output) override;

    // Get the name of the layer
    std::string get_name() const override { return "GRU"; }

    std::string get_details() const override {
        return "Hidden Size: " + std::to_string(hidden_state.size()) + "\n" +
               "Output Size: " + std::to_string(biases[3].size()) + "\n" +
               "Activation: " + activation->get_name() + "\n" +
               "Gate Activation: " + gate_activation->get_name() + "\n";
    }

    bool has_weights() const override { return true; }
    bool has_bias() const override { return true; }

    std::vector<Eigen::MatrixXf*> get_weights() override { return get_pointers(weights); }
    std::vector<Eigen::MatrixXf*> get_grad_weights() override { return get_pointers(grad_weights); }

    std::vector<Eigen::VectorXf*> get_bias() override { return get_pointers(biases); }
    std::vector<Eigen::VectorXf*> get_grad_bias() override { return get_pointers(grad_biases); }

    void serialize(std::ofstream& toFileStream) const override;
    static GRULayer* deserialize(std::ifstream& fromFileStream);

protected:
    template <typename T>
    std::vector<T*> get_pointers(std::vector<T>& vec) {
        std::vector<T*> pointers;
        for (auto& item : vec) {
            pointers.push_back(&item);
        }
        return pointers;
    }

    ~GRULayer() = default;
};

#endif // GRU_LAYER_HPP