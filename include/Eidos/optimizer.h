#ifndef OPTIMIZER_H
#define OPTIMIZER_H

#include <Eigen/Dense>
#include "layer.h"
#include <unordered_map>

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
     * @brief Pure virtual function to optimize a given layer.
     * 
     * This function is intended to be overridden by derived classes to implement
     * specific optimization algorithms. It works directly on Layer objects to 
     * adjust their parameters based on the optimization strategy.
     * 
     * @param layer Reference to the Layer object to be optimized.
     */
    virtual void optimize(Layer& layer) = 0; // Works directly on Layer objects.

    /**
     * @brief Get the name of the optimizer.
     * 
     * @return std::string The name of the optimizer.
     */
    virtual std::string get_name() const = 0;

    /**
     * @brief Serialize the optimizer to a file.
     * 
     * This function serializes the optimizer to a file stream. It is intended to be
     * overridden by derived classes to implement serialization of optimizer-specific
     * parameters.
     * 
     * @param toFileStream Reference to the output file stream.
     */
    virtual void serialize(std::ofstream& toFileStream) const = 0;

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

    void optimize(Layer& layer) override;

    std::string get_name() const override { return "SGD"; }

    void serialize(std::ofstream& toFileStream) const override {
        toFileStream.write(reinterpret_cast<const char*>(&learning_rate), sizeof(float));
    }

    static SGD* deserialize(std::ifstream& fromFileStream) {
        float learning_rate;
        fromFileStream.read(reinterpret_cast<char*>(&learning_rate), sizeof(float));
        return new SGD(learning_rate);
    }
};

class Adam : public Optimizer {
public:
    Adam(float learning_rate, float beta1 = 0.9, float beta2 = 0.999, float epsilon = 1e-8)
    : learning_rate(learning_rate), beta1(beta1), beta2(beta2), epsilon(epsilon), t(0) {};

    void optimize(Layer& layer) override;

    std::string get_name() const override { return "Adam"; }

    void serialize(std::ofstream& toFileStream) const override;

    static Adam* deserialize(std::ifstream& fromFileStream);

private:
    struct Moments {
        std::vector<Eigen::MatrixXf> m_w;  // First moment estimates for weights
        std::vector<Eigen::MatrixXf> v_w;  // Second moment estimates for weights
        std::vector<Eigen::VectorXf> m_b;  // First moment estimates for biases
        std::vector<Eigen::VectorXf> v_b;  // Second moment estimates for biases
    };

    std::unordered_map<Layer*, Moments> moments;  // Maps layers to their moment estimates
    float learning_rate, beta1, beta2, epsilon;
    int t;  // Time step

    Moments initialize_moments(Layer& layer);
};

#endif // OPTIMIZER_H
