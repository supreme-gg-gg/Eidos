#ifndef ACTIVATIONS_H
#define ACTIVATIONS_H

#include <Eigen/Dense>

/**
 * @class Activation
 * @brief Abstract base class for activation functions.
 *
 * This class provides the interface for activation functions used in neural networks.
 * Derived classes must implement the `forward` and `backward` methods.
 */
class Activation {
    public: 
    /**
     * @brief Perform the forward pass of the activation function.
     * @param input The input matrix.
     * @return The result of applying the activation function to the input.
     */
    virtual Eigen::MatrixXf forward(const Eigen::MatrixXf& input) = 0;

    /**
     * @brief Perform the backward pass of the activation function.
     * @param grad_output The gradient of the loss with respect to the output of the activation function.
     * @return The gradient of the loss with respect to the input of the activation function.
     */
    virtual Eigen::MatrixXf backward(const Eigen::MatrixXf& grad_output) = 0;
};

#endif //ACTIVATIONS_H