#ifndef LAYER_H
#define LAYER_H

#include <Eigen/Dense>
#include "tensor.hpp"
#include <fstream>

/**
 * @class Layer
 * @brief Abstract base class for neural network layers.
 * 
 * This class defines the interface for all neural network layers. It includes
 * pure virtual functions for forward and backward propagation, which must be
 * overridden by derived classes. It also provides virtual functions for checking
 * if the layer has weights or bias terms, and for retrieving these parameters.
 * 
 * Derived classes should override the necessary functions to implement specific
 * layer behavior.
 */
class Layer {
public:

    /*
     * Pure virtual function to perform forward propagation on input.
     * Must be overridden in derived class
     * 
     * @param input: The input matrix to the layer (typically activations from previous layer)
     * @return: The output matrix after applying the layer's transformation
    */
    virtual Tensor forward(const Tensor& input) = 0;

    /*
     * Pure virtual function to perform backward propagation on input.
     * Must be overridden in derived class
     * 
     * @param grad_output: The gradient of the loss function w.r.t. the output of this layer
     * @return: The gradient of the loss function w.r.t. the input of this layer
    */
    virtual Tensor backward(const Tensor& grad_output) = 0;

    /**
     * @brief Checks if the layer has weights.
     * 
     * This function returns false by default, indicating that the layer does not have weights.
     * Derived classes should override this function if they contain weights.
     * 
     * @return true if the layer has weights, false otherwise.
     */
    virtual bool has_weights() const { return false; }

    /**
     * @brief Checks if the layer has a bias term.
     * 
     * This function returns a boolean value indicating whether the layer
     * includes a bias term. By default, this function returns false, 
     * indicating that the layer does not have a bias term.
     * 
     * @return true if the layer has a bias term, false otherwise.
     */
    virtual bool has_bias() const { return false; }

    
    /**
     * @brief Retrieves the weights of the layer.
     * 
     * This function returns a vector of pointers to Eigen::MatrixXf objects,
     * representing the weights of the layer. By default, it returns an empty vector.
     * 
     * @return std::vector<Eigen::MatrixXf*> A vector of pointers to the layer's weights.
     */
    virtual std::vector<Eigen::MatrixXf*> get_weights() { return {}; }
    virtual std::vector<Eigen::MatrixXf*> get_grad_weights() { return {}; }
    
    /**
     * @brief Retrieves the biases of the layer.
     * 
     * This function returns a vector of pointers to Eigen::VectorXf objects,
     * each representing the bias of a particular neuron or unit in the layer.
     * 
     * @return std::vector<Eigen::VectorXf*> A vector of pointers to the biases.
     */
    virtual std::vector<Eigen::VectorXf*> get_bias() { return {}; }
    virtual std::vector<Eigen::VectorXf*> get_grad_bias() { return {}; }

    /**
     * @brief Sets the training mode for the layer.
     * 
     * This function is used to enable or disable the training mode for the layer.
     * When training mode is enabled, the layer may behave differently, such as
     * updating internal states or parameters.
     * 
     * @param training A boolean value indicating whether to enable (true) or 
     * disable (false) training mode.
     * 
     * @note Should be overridden if the layer has different behavior during training and inference
     */
    virtual void set_training(bool training) {}

    /**
     * @brief Get the name of the layer.
     * 
     * @return A string representing the name of the layer.
     */
    virtual std::string get_name() const { return "Layer"; }

    /**
     * @brief Serializes the layer to the given output file stream.
     * 
     * This pure virtual function must be implemented by derived classes to
     * serialize their specific data to the provided file stream.
     * 
     * @param toFileStream The output file stream to which the layer data will be serialized.
     */
    virtual void serialize(std::ofstream& toFileStream) const = 0;
    
    // Virtual destructor to allow proper cleanup of derived classes
    virtual ~Layer() = default;
};

#endif //LAYER_H