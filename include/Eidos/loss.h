#ifndef LOSS_H
#define LOSS_H

#include <Eigen/Dense>
#include "tensor.hpp"

/**
 * @class Loss
 * @brief Abstract base class for loss functions.
 * 
 * The Loss class is an abstract base class for loss functions used in training
 * machine learning models. It provides methods for computing the loss and its
 * gradient with respect to the model's predictions.
 */
class Loss {
protected:

    Eigen::MatrixXf predictions;
    Eigen::MatrixXf targets;

public:
    virtual ~Loss() = default;

    /**
     * @brief Computes the loss given the predicted and true values.
     * 
     * @param predictions The predicted values from the model (y_pred).
     * @param targets The true values (y_true).
     * @return The computed loss (scalar).
     */
    virtual float forwardMatrix(const Eigen::MatrixXf& predictions, const Eigen::MatrixXf& targets) = 0;

    /**
     * @brief Loss forward wrapper for tensors. 
     * 
     * @param predictions The predicted values from the model.
     * @param targets The actual target values.
     * @return The computed loss value as a float.
     */
    virtual float forward(const Tensor &predictions, const Tensor &targets) {
        return this->forwardMatrix(predictions.getSingleMatrix(), targets.getSingleMatrix());
    }
    
    /**
     * @brief Computes the gradient of the loss with respect to the predictions.
     * 
     * @return The gradient of the loss with respect to predictions.
     */
    virtual Eigen::MatrixXf backwardMatrix() const = 0;

    /**
     * @brief Backward pass wrapper for tensors. 
     *
     * @return A tensor containing the gradients of the loss function with respect to the predictions.
     */
    virtual Tensor backward() {
        return Tensor(this->backwardMatrix());
    }

    /**
     * @brief Get the name of the loss function.
     * 
     * @return The name of the loss function as a string.
     */
    virtual std::string get_name() const = 0;
};

#endif // LOSS_H