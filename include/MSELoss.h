#ifndef MSE_LOSS_H
#define MSE_LOSS_H

#include "loss.h"
#include <Eigen/Dense>

class MSELoss: public Loss {
public:
   float compute_loss(const Eigen::MatrixXf& predictions, const Eigen::MatrixXf& targets) const override {
        // Mean squared error: 1/n * sum((y_pred - y_true)^2)
        Eigen::MatrixXf diff = predictions - targets;
        return (diff.array().square().sum()) / predictions.rows();
   } 

   
};

#endif // MSE_LOSS_H

