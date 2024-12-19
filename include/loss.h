#ifndef LOSS_H
#define LOSS_H

#include <Eigen/Dense>

/**
 * @class Loss
 * @brief Abstract base class for loss functions.
 * 
 * The Loss class is an abstract base class for loss functions used in training
 * machine learning models. It provides methods for computing the loss and its
 * gradient with respect to the model's predictions.
 */
class Loss {
public:
    virtual ~Loss() = default;

    Eigen::MatrixXf predictions;
    Eigen::MatrixXf targets;

    /**
     * @brief Computes the loss given the predicted and true values.
     * 
     * @param predictions The predicted values from the model (y_pred).
     * @param targets The true values (y_true).
     * @return The computed loss (scalar).
     */
    virtual float forward(const Eigen::MatrixXf& predictions, const Eigen::MatrixXf& targets) = 0;
    
    /**
     * @brief Computes the gradient of the loss with respect to the predictions.
     * 
     * @param predictions The predicted values from the model (y_pred).
     * @param targets The true values (y_true).
     * @return The gradient of the loss with respect to predictions.
     */
    virtual Eigen::MatrixXf backward() const = 0;
};  

#endif // LOSS_H