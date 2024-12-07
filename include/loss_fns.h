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
    float forward(const Eigen::MatrixXf& predictions, const Eigen::MatrixXf& targets) override; 

    /**
     * @brief Computes the backward pass of the MSE loss.
     * 
     * This function calculates the gradient of the loss with respect to the input predictions.
     * 
     * @return A matrix representing the gradient of the loss.
     */
    Eigen::MatrixXf backward() const override;
};

/**
 * @class CrossEntropyLoss
 * @brief A class that implements the cross-entropy loss function.
 * 
 * This class inherits from the Loss base class and provides an implementation
 * of the cross-entropy loss function, which is commonly used in classification
 * tasks.
 */
class CrossEntropyLoss: public Loss {
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
    float forward(const Eigen::MatrixXf& predictions, const Eigen::MatrixXf& targets) override;
    
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
    Eigen::MatrixXf backward() const override;
};

class BinaryCrossEntropyLoss: public Loss {
public:
    // Forward pass: Calculate loss for binary classification
    float forward(const Eigen::MatrixXf& predictions, const Eigen::MatrixXf& targets) override;

    // Backward pass: Compute the gradient of the loss with respect to predictions
    Eigen::MatrixXf backward() const override;
};

#endif // LOSS_FNS_H

