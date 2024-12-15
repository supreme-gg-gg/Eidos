#ifndef REGULARIZATION_H
#define REGULARIZATION_H

#include "layer.h"
#include <Eigen/Dense>
#include <random>

/**
 * @class Dropout
 * @brief Implements the Dropout regularization technique.
 *
 * The Dropout class randomly sets a fraction of input units to zero at each update during training time,
 * which helps prevent overfitting.
 * 
 * @note You should apply the Dropout layer after the activation layer in the model.
 */
class Dropout: public Layer {
private:
    float probability; ///< Probability of dropping a unit.
    Eigen::MatrixXf mask; ///< Mask matrix used to drop units.
    bool training = true; ///< Flag to indicate if the layer is in training mode.

public:
    /**
     * @brief Constructor for Dropout layer.
     * @param probability Probability of dropping a unit (default is 0.5).
     */
    explicit Dropout(float probability=0.5);

    /**
     * @brief Forward pass of the Dropout layer.
     * @param input Input matrix.
     * @return Output matrix after applying dropout.
     */
    Eigen::MatrixXf forward(const Eigen::MatrixXf& input) override;

    /**
     * @brief Backward pass of the Dropout layer.
     * @param grad_output Gradient of the loss with respect to the output.
     * @return Gradient of the loss with respect to the input.
     */
    Eigen::MatrixXf backward(const Eigen::MatrixXf& grad_output) override;

    /**
     * @brief Set the training mode for the Dropout layer.
     * @param training_ Boolean flag to set training mode.
     */
    void set_training(bool training_) override;

    /**
     * @brief Destructor for Dropout layer.
     */
    ~Dropout() = default;
};

/**
 * @class BatchNorm
 * @brief Implements the Batch Normalization technique.
 *
 * The BatchNorm class normalizes the input to have zero mean and unit variance
 * for each mini-batch, which helps accelerate training and improve the performance
 * of deep neural networks.
 * 
 * @note You should apply the BatchNorm layer before the activation layer in the model.
 */
class BatchNorm: public Layer {
private:
    Eigen::VectorXf mean, variance; ///< Mean and variance of the mini-batch.
    Eigen::VectorXf running_mean, running_variance; ///< Running mean and variance for inference.
    Eigen::MatrixXf gamma; ///< Scale parameter.
    Eigen::VectorXf beta; ///< Shift parameter.
    bool training; ///< Flag to indicate if the layer is in training mode.
    float epsilon; ///< Small constant to avoid division by zero.
    int num_features; ///< Number of features in the input.
    std::default_random_engine generator; ///< Random number generator.

    Eigen::MatrixXf normalized_input; ///< Normalized input matrix.
    Eigen::MatrixXf centered_input; ///< Centered input matrix.
    Eigen::MatrixXf grad_gamma; ///< Gradient of the scale parameter.
    Eigen::VectorXf grad_beta; ///< Gradient of the shift parameter.

public:
    /**
     * @brief Constructor for BatchNorm layer.
     * @param num_features Number of features in the input.
     * @param epsilon Small constant to avoid division by zero (default is 1e-5).
     */
    explicit BatchNorm(int num_features, float epsilon=1e-5);

    /**
     * @brief Forward pass of the BatchNorm layer.
     * @param input Input matrix.
     * @return Output matrix after applying batch normalization.
     */
    Eigen::MatrixXf forward(const Eigen::MatrixXf& input) override;

    /**
     * @brief Backward pass of the BatchNorm layer.
     * @param grad_output Gradient of the loss with respect to the output.
     * @return Gradient of the loss with respect to the input.
     */
    Eigen::MatrixXf backward(const Eigen::MatrixXf& grad_output) override;

    /**
     * @brief Set the training mode for the BatchNorm layer.
     * @param training_ Boolean flag to set training mode.
     */
    void set_training(bool training_) override;

    /**
     * @brief Check if the layer has weights.
     * @return True if the layer has weights, false otherwise.
     */
    bool has_weights() const { return true; }

    /**
     * @brief Check if the layer has bias.
     * @return True if the layer has bias, false otherwise.
     */
    bool has_bias() const { return true; }

    /**
     * @brief Get the weights of the layer.
     * @return Pointer to the weights matrix.
     */
    Eigen::MatrixXf* get_weights() { return &gamma; }

    /**
     * @brief Get the gradient of the weights.
     * @return Pointer to the gradient of the weights matrix.
     */
    Eigen::MatrixXf* get_grad_weights() { return &grad_gamma; }

    /**
     * @brief Get the bias of the layer.
     * @return Pointer to the bias vector.
     */
    Eigen::VectorXf* get_bias() { return &beta; }

    /**
     * @brief Get the gradient of the bias.
     * @return Pointer to the gradient of the bias vector.
     */
    Eigen::VectorXf* get_grad_bias() { return &grad_beta; }

    /**
     * @brief Destructor for BatchNorm layer.
     */
    ~BatchNorm() = default;
};

#endif // REGULARIZATION_H