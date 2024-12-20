#ifndef ACTIVATIONS_H
#define ACTIVATIONS_H

#include <Eigen/Dense>
#include "layer.h"
#include "tensor.hpp"

/**
 * @class Activation
 * @brief Abstract base class for activation functions in a neural network layer.
 * 
 * This class provides the interface for activation functions, which are used to introduce non-linearity
 * into the network. It inherits from the Layer class and requires the implementation of forward and 
 * backward methods.
 * 
 * @note This class cannot be instantiated directly. Instead, derive from this class and implement the 
 * pure virtual methods.
 * 
 * @var Eigen::MatrixXf Activation::cache_output
 * Cache for storing the output of the forward pass, which can be used during the backward pass.
 */
class Activation : public Layer {
protected:
    Eigen::MatrixXf cache_output;
    Tensor cache_tensor;
    
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

    /**
     * @brief Applies the forward activation function to the input tensor.
     *
     * This function iterates over each depth slice of the input tensor and applies
     * the forward activation function to each slice. The modified tensor is then returned.
     *
     * @param input The input tensor to which the activation function will be applied.
     * @return The tensor after applying the activation function.
     * 
     * @note This function modifies the input tensor in-place.
     * @note This function is not required to be overridden in derived classes.
     */
    virtual Tensor forward(const Tensor& input) override {
        cache_tensor.resize(input.depth(), input[0].rows(), input[0].cols());
        Tensor result(input.depth(), input[0].rows(), input[0].cols());
        for (int i = 0; i < input.depth(); ++i) {
            result[i] = this->forward(input[i]);
            cache_tensor[i] = cache_output;
        }
        return result;
    }

    /**
     * @brief Computes the gradient of the loss with respect to the input of the activation function.
     *
     * This function performs the backward pass of the activation function, calculating the gradient
     * of the loss with respect to the input tensor based on the gradient of the loss with respect to
     * the output tensor.
     *
     * @param grad_output A tensor containing the gradient of the loss with respect to the output of the activation function.
     * @return A tensor containing the gradient of the loss with respect to the input of the activation function.
     */
    virtual Tensor backward(const Tensor& grad_output) override {
        Tensor grad_input(grad_output.depth(), grad_output[0].rows(), grad_output[0].cols());
        for (int i = 0; i < grad_output.depth(); ++i) {
            cache_output = cache_tensor[i];
            grad_input[i] = this->backward(grad_output[i]);
        }
        return grad_input;
    }
};

#endif //ACTIVATIONS_H