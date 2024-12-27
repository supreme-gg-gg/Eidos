#ifndef LOSS_FNS_H
#define LOSS_FNS_H

#include "loss.h"
#include <Eigen/Dense>

/**
 * @class MSELoss
 * @brief Mean Squared Error Loss function class.
 * 
 * This class implements the Mean Squared Error (MSE) loss function, which is commonly used 
 * in regression tasks. It inherits from the base Loss class.
 */
class MSELoss: public Loss {
public:
    /**
     * @brief Computes the forward pass of the MSE loss.
     * 
     * This function calculates the mean squared error between the predictions and the targets.
     * 
     * @param predictions A matrix of predicted values.
     * @param targets A matrix of target values.
     * @return The computed MSE loss as a float.
     */
    float forwardMatrix(const Eigen::MatrixXf& predictions, const Eigen::MatrixXf& targets) override; 

    /**
     * @brief Computes the backward pass of the MSE loss.
     * 
     * This function calculates the gradient of the loss with respect to the input predictions.
     * 
     * @return A matrix representing the gradient of the loss.
     */
    Eigen::MatrixXf backwardMatrix() const override;

    /*  */
    std::string get_name() const override { return "MSELoss"; }
};

/**
 * @class CrossEntropyLoss
 * @brief A class that implements the cross-entropy loss function with Softmax activation.
 * 
 * The CrossEntropyLoss class is derived from the Loss base class and provides
 * methods to compute the forward and backward passes of the cross-entropy loss
 * function, which is commonly used in classification tasks. This class takes in 
 * the logits (raw output) from the model and the one-hot encoded target values.
 * 
 * @note You can manually use the Softmax activation function and the CategoricalCrossEntropyLoss
 * separately, but this class combines them for convenience.
 */
class CrossEntropyLoss: public Loss {
public:
    /**
     * @brief Computes the forward pass of the loss function.
     * 
     * @param logits The predicted values (logits) as a matrix.
     * @param targets The ground truth values as a matrix.
     * @return The computed loss as a float.
     */
    float forwardMatrix(const Eigen::MatrixXf& logits, const Eigen::MatrixXf& targets) override;

    Eigen::MatrixXf backwardMatrix() const override;

    /**
     * @brief Get the name of the loss function.
     * 
     * @return The name of the loss function as a string.
     */
    std::string get_name() const override { return "CrossEntropyLoss"; }
};

/**
 * @class CategoricalCrossEntropyLoss
 * @brief A class that implements the categorical cross-entropy loss function.
 * 
 * This class inherits from the Loss base class and provides an implementation
 * of the categorical cross-entropy loss function, which is commonly used in 
 * multiclass classification tasks.
 */
class CategoricalCrossEntropyLoss: public Loss {
public:

    /**
     * @brief Computes the cross-entropy loss between predictions and targets.
     * 
     * The cross-entropy loss is given by the equation:
     * \f[
     * L = -\frac{1}{N} \sum_{i=1}^{N} \sum_{j=1}^{C} y_{ij} \log(\hat{y}_{ij})
     * \f]
     * where \(N\) is the number of samples, \(C\) is the number of classes,
     * \(y_{ij}\) is the target value, and \(\hat{y}_{ij}\) is the predicted value.
     * 
     * @param predictions A matrix of predicted values.
     * @param targets A matrix of target values (one hot encoded)
     * @return The computed cross-entropy loss.
     */
    float forwardMatrix(const Eigen::MatrixXf& predictions, const Eigen::MatrixXf& targets) override;
    
    /**
     * @brief Computes the gradient of the cross-entropy loss with respect to the predictions.
     * 
     * The gradient of the cross-entropy loss is given by the equation:
     * \f[
     * \frac{\partial L}{\partial \hat{y}_{ij}} = -\frac{y_{ij}}{\hat{y}_{ij}}
     * \f]
     * where \(y_{ij}\) is the target value and \(\hat{y}_{ij}\) is the predicted value.
     * 
     * @param predictions A matrix of predicted values.
     * @param targets A matrix of target values (one hot encoded).
     * @return The computed gradient of the cross-entropy loss.
     */
    Eigen::MatrixXf backwardMatrix() const override;

    /**
     * @brief Get the name of the loss function.
     * 
     * @return The name of the loss function as a string.
     */
    std::string get_name() const override { return "CategoricalCrossEntropyLoss"; }
};

/**
 * @class BinaryCrossEntropyLoss
 * @brief A class to compute the binary cross-entropy loss for binary classification tasks.
 * 
 * This class provides methods to perform the forward and backward passes for the binary cross-entropy loss.
 * It inherits from the base class Loss.
 */
class BinaryCrossEntropyLoss: public Loss {
public:
    // Forward pass: Calculate loss for binary classification
    float forwardMatrix(const Eigen::MatrixXf& predictions, const Eigen::MatrixXf& targets) override;

    // Backward pass: Compute the gradient of the loss with respect to predictions
    Eigen::MatrixXf backwardMatrix() const override;

    /**
     * @brief Get the name of the loss function.
     * 
     * @return The name of the loss function as a string.
     */
    std::string get_name() const override { return "BinaryCrossEntropyLoss"; }
};

#endif // LOSS_FNS_H

