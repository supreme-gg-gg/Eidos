#ifndef CONVOLUTIONAL_LAYER_H
#define CONVOLUTIONAL_LAYER_H

#include <Eigen/Dense>
#include <vector>
#include "../layer.h"

class ConvolutionalLayer : public Layer {
private:
    std::vector<Eigen::MatrixXf> weights;    // Weights for each filter
    std::vector<Eigen::VectorXf> biases;     // Biases for each output channel
    std::vector<Eigen::MatrixXf> grad_weights; // Gradients of weights
    std::vector<Eigen::VectorXf> grad_biases;  // Gradients of biases
    std::vector<Eigen::MatrixXf> input;      // Stored input for backpropagation

    int input_channels_;  // Number of input channels (depth of input feature maps)
    int output_channels_; // Number of output channels (depth of output feature maps)
    int kernel_size_;     // Size of the convolutional kernel
    int stride_;          // Stride for the convolution
    int padding_;         // Padding for the convolution

public:
    // Constructor
    ConvolutionalLayer(int input_channels, int output_channels, 
                        int kernel_size, int stride = 1, int padding = 0);

    Eigen::MatrixXf forward(const Eigen::MatrixXf& input) override {
        // This method is not implemented and should not be called.
        throw std::runtime_error("CNN Layer does not support forward pass with single input matrix. Please provide a tensor instead.");
    }

    Eigen::MatrixXf backward(const Eigen::MatrixXf& grad_output) override {
        // This method is not implemented and should not be called.
        throw std::runtime_error("CNN Layer does not support backward pass with single input matrix. Please provide a tensor instead.");
    }

    // Forward pass
    std::vector<Eigen::MatrixXf> forward(const std::vector<Eigen::MatrixXf>& input);

    // Backward pass (gradient computation)
    std::vector<Eigen::MatrixXf> backward(const std::vector<Eigen::MatrixXf>& grad_output);

    // Getter methods
    bool has_weights() const override { return true; };
    bool has_bias() const override { return true; };

    /**
     * @brief Retrieves the weights of the RNN layer.
     * 
     * This function returns a vector of pointers to Eigen::MatrixXf objects,
     * which represent the weights of the RNN layer.
     * 
     * @return std::vector<Eigen::MatrixXf*> A vector containing pointers to the weight matrices.
     */
    std::vector<Eigen::MatrixXf*> get_weights() { return get_pointers(weights); }
    std::vector<Eigen::MatrixXf*> get_grad_weights() { return get_pointers(grad_weights); }

    /**
     * @brief Retrieves the biases of the RNN layer.
     *
     * This function returns a vector of pointers to the biases of the RNN layer.
     *
     * @return std::vector<Eigen::VectorXf*> A vector containing pointers to the biases.
     */
    std::vector<Eigen::VectorXf*> get_bias() { return get_pointers(biases); }
    std::vector<Eigen::VectorXf*> get_grad_bias() { return get_pointers(grad_biases); }

protected:
    template <typename T>
    std::vector<T*> get_pointers(std::vector<T>& vec) {
        std::vector<T*> pointers;
        for (auto& item : vec) {
            pointers.push_back(&item);
        }
        return pointers;
    }
};

#endif // CONVOLUTIONAL_LAYER_H