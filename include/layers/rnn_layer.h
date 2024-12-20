#ifndef RNN_LAYER_H
#define RNN_LAYER_H

#include <Eigen/Dense>
#include "../layer.h"
#include "../activation_fns.h"
#include "../tensor.hpp"

/**
 * @class RNNLayer
 * @brief A class representing a Recurrent Neural Network (RNN) layer.
 * 
 * The RNNLayer class inherits from the Layer class and implements a recurrent neural network layer.
 * It includes methods for forward and backward passes, as well as methods to retrieve weights and biases.
 * 
 * @details
 * The RNNLayer class stores weights, biases, and their gradients in vectors for flexibility. It also maintains
 * the current hidden state and the hidden states for each time step. The class uses an activation function
 * specified during construction and can output either the entire sequence or just the final output.
 * 
 * @param input_size The size of the input vector.
 * @param hidden_size The size of the hidden state vector.
 * @param output_size The size of the output vector.
 * @param activation Pointer to the activation function to be used in the RNN layer.
 * @param output_sequence Boolean flag indicating whether to output the entire sequence (true) or just the final output (false). Default is false.
 */
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
    /**
     * @brief Constructs an RNNLayer object.
     * 
     * @param input_size The size of the input vector.
     * @param hidden_size The size of the hidden state vector.
     * @param output_size The size of the output vector.
     * @param activation Pointer to the activation function to be used in the RNN layer.
     * @param output_sequence Boolean flag indicating whether to output the entire sequence (true) or just the final output (false). Default is false.
     */
    RNNLayer(int input_size, int hidden_size, int output_size, Activation* activation, bool output_sequence = true);

    /**
     * @brief Performs the forward pass of the RNN layer.
     * 
     * @param input The input matrix to the RNN layer.
     * @return The output matrix after applying the RNN layer.
     */
    Tensor forward(const Tensor& input) override;

    /**
     * @brief Performs the backward pass for the RNN layer.
     * 
     * This function computes the gradient of the loss with respect to the input
     * of the RNN layer, given the gradient of the loss with respect to the output
     * of the RNN layer.
     * 
     * @param grad_output The gradient of the loss with respect to the output of the RNN layer.
     * @return The gradient of the loss with respect to the input of the RNN layer.
     */
    Tensor backward(const Tensor& grad_output) override;

    bool has_weights() const override { return true; }
    bool has_bias() const override { return true; }

    /**
     * @brief Retrieves the weights of the RNN layer.
     * 
     * This function returns a vector of pointers to Eigen::MatrixXf objects,
     * which represent the weights of the RNN layer.
     * 
     * @return std::vector<Eigen::MatrixXf*> A vector containing pointers to the weight matrices.
     */
    std::vector<Eigen::MatrixXf*> get_weights() override { return get_pointers(weights); }
    std::vector<Eigen::MatrixXf*> get_grad_weights() override { return get_pointers(grad_weights); }

    /**
     * @brief Retrieves the biases of the RNN layer.
     *
     * This function returns a vector of pointers to the biases of the RNN layer.
     *
     * @return std::vector<Eigen::VectorXf*> A vector containing pointers to the biases.
     */
    std::vector<Eigen::VectorXf*> get_bias() override { return get_pointers(biases); }
    std::vector<Eigen::VectorXf*> get_grad_bias() override { return get_pointers(grad_biases); }
    
    ~RNNLayer() = default;

protected:
    /**
     * @brief Converts a vector of objects to a vector of pointers to those objects.
     * 
     * This function takes a vector of objects and returns a vector containing pointers
     * to each of the objects in the original vector.
     * 
     * @tparam T The type of the objects in the vector.
     * @param vec A reference to a vector of objects of type T.
     * @return std::vector<T*> A vector of pointers to the objects in the input vector.
     */
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