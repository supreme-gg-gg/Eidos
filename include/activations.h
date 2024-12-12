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
class Activation : public Layer<Eigen::MatrixXf> {
protected:
    Eigen::MatrixXf cache_output;
    Tensor<Eigen::MatrixXf> cache_output_tensor;
    
public: 
    /**
     * @brief Perform the forward pass of the activation function.
     * @param input The input matrix.
     * @return The result of applying the activation function to the input.
     */
    virtual Eigen::MatrixXf forward(const Eigen::MatrixXf& input) = 0;

    virtual Tensor<Eigen::MatrixXf> forward(const Tensor<Eigen::MatrixXf>& input)= 0;

    /**
     * @brief Perform the backward pass of the activation function.
     * @param grad_output The gradient of the loss with respect to the output of the activation function.
     * @return The gradient of the loss with respect to the input of the activation function.
     */
    virtual Eigen::MatrixXf backward(const Eigen::MatrixXf& grad_output) = 0;

    virtual Tensor<Eigen::MatrixXf> backward(const Tensor<Eigen::MatrixXf>& grad_output) = 0;

    /**
     * @brief Get the output of the forward pass.
     * @return The output of the forward pass.
     */
    Eigen::MatrixXf cache() const {
        return cache_output;
    }
};

#endif //ACTIVATIONS_H