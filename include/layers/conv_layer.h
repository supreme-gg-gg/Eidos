#ifndef CONVOLUTIONAL_LAYER_H
#define CONVOLUTIONAL_LAYER_H

#include <Eigen/Dense>
#include <vector>
#include "../layer.h"
#include "../tensor.hpp"

/**
 * @class Conv2D
 * @brief Represents a 2D convolutional layer in a neural network.
 *
 * This class implements a 2D convolutional layer which is a fundamental building block
 * in convolutional neural networks (CNNs). It performs convolution operations on the input data.
 *
 * @param input_channels The number of input channels (depth) of the input data.
 * @param output_channels The number of output channels (depth) of the output data.
 * @param kernel_size The size of the convolutional kernel (filter).
 * @param stride The stride of the convolution operation. Default is 1.
 * @param padding The amount of zero-padding added to the input data. Default is 0.
 */
class Conv2D: public Layer {
private:
    std::vector<Eigen::MatrixXf> weights;    // Weights for each filter
    std::vector<Eigen::VectorXf> biases;     // Biases for each output channel
    std::vector<Eigen::MatrixXf> grad_weights; // Gradients of weights
    std::vector<Eigen::VectorXf> grad_biases;  // Gradients of biases

    std::vector<int> input_shape;
    std::vector<int> output_shape;
    int kernel_size_;     // Size of the convolutional kernel
    int stride_;          // Stride for the convolution
    int padding_;         // Padding for the convolution

    Tensor cache_input;         // Cached input for backward pass

    Tensor applyPadding(const Tensor& input);
    std::vector<int> calculateOutputShape() const;

public:
    /**
     * @class ConvolutionalLayer
     * @brief Represents a convolutional layer in a neural network.
     *
     * This class implements a convolutional layer which is a fundamental building block
     * in convolutional neural networks (CNNs). It performs convolution operations on the input data.
     *
     * @param input_channels The number of input channels (depth) of the input data.
     * @param output_channels The number of output channels (depth) of the output data.
     * @param kernel_size The size of the convolutional kernel (filter).
     * @param stride The stride of the convolution operation. Default is 1.
     * @param padding The amount of zero-padding added to the input data. Default is 0.
     */
    Conv2D(int input_channels, int output_channels, 
                        int kernel_size, int stride = 1, int padding = 0);

    ~Conv2D() = default;

    // Forward pass
    Tensor forward(const Tensor& input) override;

    // Backward pass (gradient computation)
    Tensor backward(const Tensor& grad_output) override;

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
    std::vector<Eigen::MatrixXf*> get_weights() override { return get_pointers(weights); }
    std::vector<Eigen::MatrixXf*> get_grad_weights() override { return get_pointers(grad_weights); }

    /**
     * @brief Retrieves the biases of the RNN layer.
     *
     * This function returns a vector of pointers to the biases of the RNN layer.
     *
     * @return std::vector<Eigen::VectorXf*> A vector containing pointers to the biases.
     */
    std::vector<Eigen::VectorXf*> get_bias() override { return get_pointers(biases); }
    std::vector<Eigen::VectorXf*> get_grad_bias() override { return get_pointers(grad_biases); }

    std::string get_name() const override { return "Conv2D"; }

    std::string get_details() const override {
        return "   Input Shape: " + std::to_string(input_shape[0]) + "x" + std::to_string(input_shape[1]) + "x" + std::to_string(input_shape[2]) + "\n" +
               "   Output Shape: " + std::to_string(output_shape[0]) + "x" + std::to_string(output_shape[1]) + "x" + std::to_string(output_shape[2]) + "\n" +
               "   Kernel Size: " + std::to_string(kernel_size_) + "\n" +
               "   Stride: " + std::to_string(stride_) + "\n" +
               "   Padding: " + std::to_string(padding_) + "\n";
    }
    
    void serialize(std::ofstream& toFileStream) const override;
    static Conv2D* deserialize(std::ifstream& fromFileStream);

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