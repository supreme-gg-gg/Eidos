#ifndef LAYER_H
#define LAYER_H

#include <Eigen/Dense>

/*
 * Abstract base class for a neural network layer
 * Derived class should implement forward and backward methods.
*/
class Layer {
public:

    /*
     * Virtual function to perform forward propagation on input.
     * Must be overridden in derived class
     * 
     * @param input: The input matrix to the layer (typically activations from previous layer)
     * @return: The output matrix after applying the layer's transformation
    */
    virtual Eigen::MatrixXf forward(const Eigen::MatrixXf& input) = 0;

    /*
     * Virtual function to perform backward propagation on input.
     * Must be overridden in derived class
     * 
     * @param grad_output: The gradient of the loss function w.r.t. the output of this layer
     * @return: The gradient of the loss function w.r.t. the input of this layer
    */
    virtual Eigen::MatrixXf backward(const Eigen::MatrixXf& grad_output) = 0;

    virtual bool has_weights() const { return false; }
    virtual bool has_bias() const { return false; }

    virtual Eigen::MatrixXf* get_weights() { return nullptr; }
    virtual Eigen::MatrixXf* get_grad_weights() { return nullptr; }
    virtual Eigen::VectorXf* get_bias() { return nullptr; }
    virtual Eigen::VectorXf* get_grad_bias() { return nullptr; }

    // Virtual destructor to allow proper cleanup of derived classes
    virtual ~Layer() = default;
};

#endif //LAYER_H