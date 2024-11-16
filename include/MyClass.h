#ifndef MYCLASS_H
#define MYCLASS_H

#include <Eigen/Dense>

class MyClass {
public:
    Eigen::MatrixXd multiplyMatrices(const Eigen::MatrixXd &mat1, const Eigen::MatrixXd &mat2);
};

#endif
