#ifndef LAYER_H
#define LAYER_H

#include <Eigen/Dense>
#include "tensor.hpp"
#include <fstream>

/*
 * Abstract base class for a neural network layer
 * Derived class should implement forward and backward methods.
*/
class Layer {
public:

    /*
     * Pure virtual function to perform forward propagation on input.
     * Must be overridden in derived class
     * 
     * @param input: The input matrix to the layer (typically activations from previous layer)
     * @return: The output matrix after applying the layer's transformation
    */
    virtual Tensor forward(const Tensor& input) = 0;

    /*
     * Pure virtual function to perform backward propagation on input.
     * Must be overridden in derived class
     * 
     * @param grad_output: The gradient of the loss function w.r.t. the output of this layer
     * @return: The gradient of the loss function w.r.t. the input of this layer
    */
    virtual Tensor backward(const Tensor& grad_output) = 0;

    // Virtual functions that can be overridden in derived classes but are not required
    // The most derived version of the function will be called
    virtual bool has_weights() const { return false; }
    virtual bool has_bias() const { return false; }

    virtual std::vector<Eigen::MatrixXf*> get_weights() { return {}; }
    virtual std::vector<Eigen::MatrixXf*> get_grad_weights() { return {}; }
    virtual std::vector<Eigen::VectorXf*> get_bias() { return {}; }
    virtual std::vector<Eigen::VectorXf*> get_grad_bias() { return {}; }

    // Should be overridden if the layer has different behavior during training and inference
    virtual void set_training(bool training) {}

    virtual std::string get_name() const { return "Layer"; }

    virtual void serialize(std::ofstream& toFileStream) const = 0;
    
    // Virtual destructor to allow proper cleanup of derived classes
    virtual ~Layer() = default;
};

#endif //LAYER_H