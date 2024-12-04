#ifndef MSE_LOSS_H
#define MSE_LOSS_H

#include "loss.h"
#include <Eigen/Dense>

class MSELoss: public Loss {
public:
   float compute_loss(const Eigen::MatrixXf& predictions, const Eigen::MatrixXf& targets) const override; 
   Eigen::MatrixXf compute_gradient(const Eigen::MatrixXf& predictions, const Eigen::MatrixXf& targets) const override;
};

#endif // MSE_LOSS_H

