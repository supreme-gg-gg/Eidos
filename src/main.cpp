#include <iostream>
#include <Eigen/Dense>

int main() {
    Eigen::MatrixXd mat(2, 2);
    mat(0, 0) = 1.0;
    mat(0, 1) = 2.0;
    mat(1, 0) = 3.0;
    mat(1, 1) = 4.0;

    std::cout << "Matrix:\n" << mat << std::endl;
    return 0;
}
