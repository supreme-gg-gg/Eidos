#ifndef RELU_H
#define RELU_H

#include "activations.h"
#include <Eigen/Dense>

/**
 * @class ReLU
 * @brief Implements the Rectified Linear Unit (ReLU) activation function.
 *
 * The ReLU activation function is defined as:
 * \f[
 * f(x) = \max(0, x)
 * \f]
 * It sets all negative values in the input to zero and keeps positive values unchanged.
 */
class ReLU: public Activation {
    public:
    /**
     * @brief Applies the ReLU activation function to the input matrix.
     * 
     * This function performs an element-wise maximum operation between the input matrix
     * and 0, effectively setting all negative values in the input matrix to 0.
     * 
     * @param input The input matrix to which the ReLU activation function will be applied.
     * @return Eigen::MatrixXf The resulting matrix after applying the ReLU activation function.
     */
    Eigen::MatrixXf forward(const Eigen::MatrixXf& input) override {
        return input.cwiseMax(0); // Element-wise max with 0
    }

    /**
     * @brief Computes the gradient of the ReLU activation function.
     * 
     * This function performs an element-wise multiplication between the gradient of the output
     * and a binary matrix indicating where the original input was greater than 0.
     * 
     * @param grad_output The gradient of the loss with respect to the output of the ReLU function.
     * @return Eigen::MatrixXf The gradient of the loss with respect to the input of the ReLU function.
     */
    Eigen::MatrixXf backward(const Eigen::MatrixXf& grad_output) override {
        return grad_output.cwiseProduct((grad_output.array() > 0).cast<float>());
    }
};

#endif //RELU_H 