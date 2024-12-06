#ifndef ACTIVATION_FNS_H
#define ACTIVATION_FNS_H

#include "activations.h"
#include "layer.h"
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
class ReLU : public Layer {
private:
    Eigen::MatrixXf mask;

public:
    Eigen::MatrixXf forward(const Eigen::MatrixXf& input) override {
        mask = (input.array() > 0).cast<float>();
        return input.cwiseMax(0); // ReLU activation
    }

    Eigen::MatrixXf backward(const Eigen::MatrixXf& grad_output) override {
        return grad_output.cwiseProduct(mask); // Grad of ReLU
    }
};

/**
 * @class Sigmoid
 * @brief Implements the Sigmoid activation function.
 *
 * The Sigmoid activation function is defined as:
 * \f[
 * f(x) = \frac{1}{1 + e^{-x}}
 * \f]
 * It squashes the input values to the range (0, 1).
 */
class Sigmoid: public Activation {
    public: 
    /**
     * @brief Applies the Sigmoid activation function to the input matrix.
     * 
     * This function applies the Sigmoid activation function to the input matrix element-wise.
     * 
     * @param input The input matrix to which the Sigmoid activation function will be applied.
     * @return Eigen::MatrixXf The resulting matrix after applying the Sigmoid activation function.
     */
    Eigen::MatrixXf forward(const Eigen::MatrixXf& input) override {
        cache_output = 1.0f / (1.0f + (-input.array()).exp());
        return cache_output;
    }

    /**
     * @brief Computes the gradient of the Sigmoid activation function.
     * 
     * This function computes the gradient of the Sigmoid activation function using the output
     * of the Sigmoid function. The gradient is computed as the element-wise product of the
     * gradient of the output and the element-wise product of the output and 1 - output.
     * 
     * @param grad_output The gradient of the loss with respect to the output of the Sigmoid function.
     * @return Eigen::MatrixXf The gradient of the loss with respect to the input of the Sigmoid function.
     */
    Eigen::MatrixXf backward(const Eigen::MatrixXf& grad_output) override {
        Eigen::MatrixXf sigmoid_grad = cache_output.array() * (1.0f - cache_output.array());
        return grad_output.array() * sigmoid_grad.array();
    }

    private:
    Eigen::MatrixXf cache_output; // Cache the output of the Sigmoid function
};


/**
 * @class Softmax
 * @brief A class that implements the Softmax activation function.
 * 
 * The Softmax class provides methods to apply the Softmax activation function to an input matrix
 * and to compute the gradient of the Softmax activation function.
 */
class Softmax: public Activation {
    public:
    /**
     * @brief Applies the Softmax activation function to the input matrix.
     * 
     * This function applies the Softmax activation function to the input matrix element-wise.
     * 
     * @param input The input matrix to which the Softmax activation function will be applied.
     * @return Eigen::MatrixXf The resulting matrix after applying the Softmax activation function.
     */
    Eigen::MatrixXf forward(const Eigen::MatrixXf& logits) override {
        Eigen::MatrixXf exp_logits = (logits.array() - logits.rowwise().maxCoeff().array()).exp(); // Subtract max for numerical stability
        Eigen::MatrixXf probabilities = exp_logits.array().colwise() / exp_logits.rowwise().sum().array(); // Normalize by sum
        cache_output = probabilities; // Cache the output for backpropagation
        return probabilities;
    }

    /**
     * @brief Computes the gradient of the Softmax activation function.
     * 
     * This function computes the gradient of the Softmax activation function using the output
     * of the Softmax function. The gradient is computed as the element-wise product of the
     * gradient of the output and the element-wise product of the output and 1 - output.
     * 
     * @param grad_output The gradient of the loss with respect to the output of the Softmax function.
     * @return Eigen::MatrixXf The gradient of the loss with respect to the input of the Softmax function.
     */
    Eigen::MatrixXf backward(const Eigen::MatrixXf& grad_output) override {
        Eigen::MatrixXf grad = cache_output;
        for (int i = 0; i < grad.rows(); ++i) {
            Eigen::MatrixXf jacobian = grad.row(i).transpose() * grad.row(i); // Jacobian
            jacobian.diagonal() -= grad.row(i).transpose(); // subtract diagonal
            grad.row(i) = grad_output.row(i) * jacobian; // chain rule
        }
        return grad;
    }

    private:
    Eigen::MatrixXf cache_output; // Cache the output of the Softmax function
};

#endif //ACTIVATION_FNS_H