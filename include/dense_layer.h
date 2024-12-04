#ifndef DENSE_LAYER_H
#define DENSE_LAYER_H

#include "layer.h"
#include "activations.h"
#include <Eigen/Dense>

/**
 * @class DenseLayer
 * 
 * 此層為重重疊疊之結構，猶如山川重峦，層層疊起，
 * 設計精巧，層層遞進，每一層皆重若千鈞。
 * 
 * 其內部運算，猶如群山之間，流轉無窮，權重與偏置交織，
 * 轉換輸入，成就最終之輸出。
 */
class DenseLayer: public Layer {
    private: 
        Eigen::MatrixXf weights; ///< Weight matrix (W)
        Eigen::VectorXf bias; ///< Bias vector (b)
        Eigen::MatrixXf input; ///< Input matrix (x)
        Eigen::MatrixXf output; ///< Output matrix (y)

        Eigen::MatrixXf grad_weights; ///< Gradient of the loss w.r.t. the weights
        Eigen::VectorXf grad_bias; ///< Gradient of the loss w.r.t. the bias

        Activation* activation; ///< Activation function for the layer
    public:
        /**
         * @brief Constructor for the DenseLayer class.
         * 
         * @param input_size The size of the input vector.
         * @param output_size The size of the output vector.
         * @param activation The activation function to use in the layer.
         */
        DenseLayer(int input_size, int output_size, Activation* activation);

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
        Eigen::VectorXf get_bias_gradient() const;
};

#endif //DENSE_LAYER_H