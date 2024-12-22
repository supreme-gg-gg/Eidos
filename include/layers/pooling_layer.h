#ifndef POOLING_LAYER_H
#define POOLING_LAYER_H

#include <Eigen/Dense>
#include "../layer.h"

/**
 * @class MaxPooling2D
 * @brief A class representing a 2D max pooling layer in a neural network.
 * 
 * This class implements a 2D max pooling layer, which reduces the spatial dimensions
 * of the input tensor by taking the maximum value over a specified window size.
 * 
 * @details
 * The MaxPooling2D layer is commonly used in convolutional neural networks (CNNs)
 * to downsample the input, reducing the number of parameters and computation in the network.
 * It also helps to make the network invariant to small translations of the input.
 */
class MaxPooling2D: public Layer {
public:
    /**
     * @brief Constructs a MaxPooling2D layer.
     * 
     * @param pool_size The size of the pooling window.
     * @param stride The stride of the pooling operation.
     */
    MaxPooling2D(int pool_size, int stride);

    Tensor forward(const Tensor& input) override;

    Tensor backward(const Tensor& grad_output) override;

    std::string get_name() const override { return "MaxPooling2D"; }

private:
    int pool_size;
    int stride;
    std::vector<int> input_shape;
    std::vector<int> output_shape;
    Tensor input;
    Tensor mask;
};

/**
 * @class AveragePooling2D
 * @brief Implements a 2D average pooling layer.
 *
 * This class performs average pooling operation on the input tensor.
 * It reduces the spatial dimensions (height and width) of the input tensor
 * by taking the average value within each pooling window.
 */
class AveragePooling2D: public Layer {
public:

    /**
     * @brief Constructs an AveragePooling2D layer.
     * 
     * @param pool_size Size of the pooling window.
     * @param stride Stride of the pooling operation.
     */
    AveragePooling2D(int pool_size, int stride);

    Tensor forward(const Tensor& input) override;

    Tensor backward(const Tensor& grad_output) override;

    std::string get_name() const override { return "AveragePooling2D"; }

private:
    int pool_size;
    int stride;
    std::vector<int> input_shape;
    std::vector<int> output_shape;
    Tensor input;
    Tensor mask;
};

#endif // POOLING_LAYER_H