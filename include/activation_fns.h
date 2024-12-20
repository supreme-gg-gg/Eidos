#ifndef ACTIVATION_FNS_H
#define ACTIVATION_FNS_H

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
class ReLU : public Activation {
public:
    /**
     * @brief Applies the forward pass of the activation function to the input matrix.
     *
     * @param input The input matrix of size (m x n) where m is the number of samples and n is the number of features.
     * @return Eigen::MatrixXf The output matrix after applying the activation function.
     */
    Eigen::MatrixXf forward(const Eigen::MatrixXf& input) override;

    /**
     * @brief Computes the gradient of the loss with respect to the input of the activation function.
     *
     * This function takes the gradient of the loss with respect to the output of the activation function
     * and computes the gradient of the loss with respect to the input of the activation function.
     *
     * @param grad_output The gradient of the loss with respect to the output of the activation function.
     * @return The gradient of the loss with respect to the input of the activation function.
     */
    Eigen::MatrixXf backward(const Eigen::MatrixXf& grad_output) override;
};

/**
 * @class LeakyReLU
 * @brief Implements the Leaky Rectified Linear Unit (LeakyReLU) activation function.
 *
 * The LeakyReLU activation function is defined as:
 * \f[
 * f(x) = \begin{cases} 
 * x & \text{if } x > 0 \\
 * \alpha x & \text{otherwise}
 * \end{cases}
 * \f]
 * It allows a small, non-zero gradient when the input is negative, controlled by the parameter \f$\alpha\f$.
 */
class LeakyReLU : public Activation {
public:
    LeakyReLU(float alpha = 0.01) : alpha(alpha) {}

    /**
     * @brief Performs the forward pass of the activation function
     * 
     * @param input The input matrix to apply the activation function to
     * @return Eigen::MatrixXf The result of applying the activation function element-wise
     * 
     * This is a virtual function that must be implemented by derived classes.
     * The function applies the activation function to each element of the input matrix.
     */
    Eigen::MatrixXf forward(const Eigen::MatrixXf& input) override;

    /**
     * @brief Computes the backward pass (gradient) of the activation function
     * 
     * @param grad_output The gradient from the next layer (dL/dY)
     * @return Eigen::MatrixXf The gradient with respect to the input (dL/dX)
     */
    Eigen::MatrixXf backward(const Eigen::MatrixXf& grad_output) override;

private:
    float alpha;  // The slope for negative values
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
    Eigen::MatrixXf forward(const Eigen::MatrixXf& input) override;

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
    Eigen::MatrixXf backward(const Eigen::MatrixXf& grad_output) override;
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
    Eigen::MatrixXf forward(const Eigen::MatrixXf& logits) override;

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
    Eigen::MatrixXf backward(const Eigen::MatrixXf& grad_output) override;
};

/**
 * @class Tanh
 * @brief Implements the Tanh activation function.
 * 
 * The Tanh activation function is defined as:
 * \f[
 * \text{Tanh}(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}
 * \f]
 * 
 * This class provides methods to apply the Tanh activation function to an input matrix
 * and to compute the gradient of the Tanh activation function.
 * 
 * @note The Tanh activation function squashes the input values to be between -1 and 1.
 */
class Tanh: public Activation {
    public:
    /**
     * @brief Applies the Tanh activation function to the input matrix.
     * 
     * This function applies the Tanh activation function to the input matrix element-wise.
     * 
     * @param input The input matrix to which the Tanh activation function will be applied.
     * @return Eigen::MatrixXf The resulting matrix after applying the Tanh activation function.
     */
    Eigen::MatrixXf forward(const Eigen::MatrixXf& input) override;

    /**
     * @brief Computes the gradient of the Tanh activation function.
     * 
     * This function computes the gradient of the Tanh activation function using the output
     * of the Tanh function. The gradient is computed as the element-wise product of the
     * gradient of the output and the element-wise product of the output and 1 - output.
     * 
     * @param grad_output The gradient of the loss with respect to the output of the Tanh function.
     * @return Eigen::MatrixXf The gradient of the loss with respect to the input of the Tanh function.
     */
    Eigen::MatrixXf backward(const Eigen::MatrixXf& grad_output) override;

};

#endif //ACTIVATION_FNS_H