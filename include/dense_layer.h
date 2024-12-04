#ifndef DENSE_LAYER_H
#define DENSE_LAYER_H

#include "layer.h"
#include <Eigen/Dense>

/**
 * @class DenseLayer
 * @brief A class representing a dense (fully connected) layer in a neural network.
 * 
 * The DenseLayer class inherits from the Layer class and implements the forward and backward
 * passes for a dense layer. It uses the Eigen library for matrix operations.
 */
class DenseLayer: public Layer {
    private: 
        Eigen::MatrixXf weights; ///< Weight matrix (W)
        Eigen::MatrixXf bias; ///< Bias vector (b)
        Eigen::MatrixXf input; ///< Input matrix (x)

        Eigen::MatrixXf grad_weights; ///< Gradient of the loss w.r.t. the weights
        Eigen::MatrixXf grad_bias; ///< Gradient of the loss w.r.t. the bias
    public:
        /**
         * @brief Constructor for the DenseLayer class.
         * 
         * @param input_size The size of the input vector.
         * @param output_size The size of the output vector.
         */
        DenseLayer(int input_size, int output_size);

        /**
         * @brief Performs the forward pass of the dense layer.
         * 
         * @param input The input matrix (x).
         * @return The output matrix (y) after applying the linear transformation.
         */
        Eigen::MatrixXf forward(const Eigen::MatrixXf& input) override;

        /**
         * @brief Performs the backward pass of the dense layer.
         * 
         * @param grad_output The gradient of the loss with respect to the output (dy).
         * @return The gradient of the loss with respect to the input (dx).
         */
        Eigen::MatrixXf backward(const Eigen::MatrixXf& grad_output) override;

        /**
         * @brief Gets the gradient of the loss with respect to the weights.
         * 
         * @return The gradient of the loss with respect to the weights.
         */
        Eigen::MatrixXf get_weights_gradient() const;

        /**
         * @brief Gets the gradient of the loss with respect to the bias.
         * 
         * @return The gradient of the loss with respect to the bias.
         */
        Eigen::MatrixXf get_bias_gradient() const;
};

#endif //DENSE_LAYER_H