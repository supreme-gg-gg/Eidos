#ifndef LOSS_H
#define LOSS_H

#include <Eigen/Dense>

class Loss {
public:
    virtual ~Loss() = default;

    /**
     * @brief Computes the loss given the predicted and true values.
     * 
     * @param predictions The predicted values from the model (y_pred).
     * @param targets The true values (y_true).
     * @return The computed loss (scalar).
     */
    virtual float compute_loss(const Eigen::MatrixXf& predictions, const Eigen::MatrixXf& targets) const = 0;
    
    /**
     * @brief Computes the gradient of the loss with respect to the predictions.
     * 
     * @param predictions The predicted values from the model (y_pred).
     * @param targets The true values (y_true).
     * @return The gradient of the loss with respect to predictions.
     */
    virtual Eigen::MatrixXf compute_gradient(const Eigen::MatrixXf& predictions, const Eigen::MatrixXf& targets) const = 0;
};  

#endif // LOSS_H