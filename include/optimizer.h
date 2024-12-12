#ifndef OPTIMIZER_H
#define OPTIMIZER_H

#include <Eigen/Dense>
#include "layer.h"
#include "tensor.h"
#include <unordered_map>

/**
 * @class Optimizer
 * @brief Abstract class for optimization algorithms.
 * 
 * This class defines the interface for optimization algorithms. All optimization algorithms
 * should inherit from this class and implement the `optimize` method. 
 * 
 * We implement two versions of the `optimize` method, one for layers using MatrixXf as the data type
 * and another for layers using Tensor<MatrixXf> as the data type. This way we can use the same optimizer
 * for both types of layers by overloading the appropriate method using polymorphism.
 */
class Optimizer {
    public:

    /**
     * @brief Pure virtual function to optimize a given layer.
     * 
     * This is reserved for layers using MatrixXf as the data type.
     * 
     * This function is intended to be overridden by derived classes to implement
     * specific optimization algorithms. It works directly on Layer objects to 
     * adjust their parameters based on the optimization strategy.
     * 
     * @param layer Reference to the Layer object to be optimized.
     */
    virtual void optimize(Layer<Eigen::MatrixXf>& layer) = 0; // Works directly on Layer objects.

    /**
     * @brief Pure virtual function to optimize a given layer.
     * 
     * This is reserved for layers using Tensor<MatrixXf> as the data type.
     * 
     * This function is intended to be overridden by derived classes to implement
     * specific optimization algorithms. It works directly on Layer objects to 
     * adjust their parameters based on the optimization strategy.
     * 
     * @param layer Reference to the Layer object to be optimized.
     */
    virtual void optimizer(Layer<Tensor<Eigen::MatrixXf>>& layer) = 0;

    virtual ~Optimizer() = default;
};


class SGD: public Optimizer {
private: 
    float learning_rate;

public:
    
    /**
     * @brief Constructs a new SGD optimizer with a given learning rate.
     * 
     * @param learning_rate The learning rate of the optimizer.
     */
    explicit SGD(float learning_rate): learning_rate(learning_rate) {}

    void optimize(Layer<Eigen::MatrixXf>& layer) override;

    void optimize(Layer<Tensor<Eigen::MatrixXf>>& layer) override;
};
class Adam : public Optimizer {
public:
    Adam(float learning_rate, float beta1 = 0.9, float beta2 = 0.999, float epsilon = 1e-8)
    : learning_rate(learning_rate), beta1(beta1), beta2(beta2), epsilon(epsilon), t(0) {};

    void optimize(Layer<Eigen::MatrixXf>& layer) override;

    void optimize(Layer<Tensor<Eigen::MatrixXf>>& layer) override;

private:
    struct Moments {
        Eigen::MatrixXf m_w;   // First moment estimate for weights
        Eigen::MatrixXf v_w;   // Second moment estimate for weights
        Eigen::VectorXf m_b;   // First moment estimate for bias
        Eigen::VectorXf v_b;   // Second moment estimate for bias
    };

    // void* is a pointer to either Layer<Eigen::MatrixXf> or Layer<Tensor<Eigen::MatrixXf>>
    std::unordered_map<void*, Moments> moments;  // Maps layers to their moment estimates
    float learning_rate, beta1, beta2, epsilon;
    int t;  // Time step

    Moments initialize_moments(Layer<Eigen::MatrixXf>& layer); 
    Moments initialize_moments(Layer<Tensor<Eigen::MatrixXf>>& layer);

    template <typename LayerType>
    void optimize_impl(LayerType& layer);
};

#endif // OPTIMIZER_H
