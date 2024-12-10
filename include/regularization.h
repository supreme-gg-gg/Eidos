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
 * @brief Placeholder for BatchNormalization class.
 *
 * The BatchNormalization class will implement the batch normalization technique.
 */
class BatchNorm: public Layer {
private:
    Eigen::MatrixXf mean, variance;
    Eigen::MatrixXf running_mean, running_variance;
    Eigen::MatrixXf gamma, beta;
    bool training;
    float epsilon;
    int num_features;
    std::default_random_engine generator;

    Eigen::MatrixXf normalized_input;
    Eigen::MatrixXf centered_input;
    Eigen::MatrixXf grad_gamma;
    Eigen::MatrixXf grad_beta;

public:
    explicit BatchNorm(int num_features, float epsilon=1e-5);

    Eigen::MatrixXf forward(const Eigen::MatrixXf& input) override;

    Eigen::MatrixXf backward(const Eigen::MatrixXf& grad_output) override;

    void set_training(bool training_) override;

    ~BatchNorm() = default;
};

#endif // REGULARIZATION_H