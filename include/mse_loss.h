#ifndef MSE_LOSS_H
#define MSE_LOSS_H

#include "loss.h"
#include <Eigen/Dense>

class MSELoss: public Loss {
public:
   float forward(const Eigen::MatrixXf& predictions, const Eigen::MatrixXf& targets) override; 
   Eigen::MatrixXf backward() const override;
};

#endif // MSE_LOSS_H

