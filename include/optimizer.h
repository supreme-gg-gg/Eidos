#ifndef OPTIMIZER_H
#define OPTIMIZER_H

#include <Eigen/Dense>

/**
 * @class Optimizer
 * @brief Abstract class for optimization algorithms.
 * 
 * This class defines the interface for optimization algorithms. All optimization algorithms
 * should inherit from this class and implement the `optimize` method.
 */
class Optimizer {
    public:
    /**
     * @brief Pure virtual function to update weights and biases.
     * 
     * This function updates the weights and biases of a neural network layer
     * based on the provided gradients. It must be implemented by any derived
     * optimizer class.
     * 
     * @param weights Reference to the matrix of weights to be updated.
     * @param weight_gradients Constant reference to the matrix of weight gradients.
     * @param bias Pointer to the vector of biases to be updated. Can be nullptr if no biases are used.
     * @param bias_gradients Constant pointer to the vector of bias gradients. Can be nullptr if no biases are used.
     */
    virtual void update(Eigen::MatrixXf& weights, const Eigen::MatrixXf& weight_gradients,
                        Eigen::VectorXf* bias, const Eigen::VectorXf* bias_gradients) = 0;
};


class SGD: public Optimizer {
    public:
    float learning_rate;

    /**
     * @brief Constructs a new SGD optimizer with a given learning rate.
     * 
     * @param learning_rate The learning rate of the optimizer.
     */
    SGD(float learning_rate): learning_rate(learning_rate) {}

    void update(Eigen::MatrixXf& weights, const Eigen::MatrixXf& weight_gradients,
                Eigen::VectorXf* bias, const Eigen::VectorXf* bias_gradients) override;
};

class Adam: public Optimizer {
    public:
    float learning_rate;
    float beta1;
    float beta2;
    float epsilon;
    Eigen::MatrixXf m; // First moment estimate
    Eigen::MatrixXf v; // Second moment estimate
    Eigen::VectorXf m_bias; // First moment estimate for bias
    Eigen::VectorXf v_bias; // Second moment estimate for bias
    int t; // Time update

    Adam(float lr, float beta1 = 0.9, float beta2 = 0.999, float epsilon = 1e-8)
        : learning_rate(lr), beta1(beta1), beta2(beta2), epsilon(epsilon), t(0) {}

    void update(Eigen::MatrixXf& weights, const Eigen::MatrixXf& weight_gradients,
                Eigen::VectorXf* bias, const Eigen::VectorXf* bias_gradients) override;
};

#endif // OPTIMIZER_H
