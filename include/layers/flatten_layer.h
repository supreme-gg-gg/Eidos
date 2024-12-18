#ifndef FLATTEN_LAYER_H
#define FLATTEN_LAYER_H

#include <Eigen/Dense>
#include <vector>
#include <stdexcept>
#include "../layer.h"

/**
 * @class FlattenLayer
 * @brief A class representing a Flatten layer.
 * 
 * The FlattenLayer class inherits from the Layer class and implements a layer that flattens the input.
 * It includes methods for forward and backward passes.
 */
class FlattenLayer : public Layer {
private:
    std::vector<int> input_shape;  // Input dimensions (e.g., {Channels, Height, Width})
    std::vector<int> output_shape; // Flattened dimensions (e.g., {Channels, Flattened_Size})

public:
    // Constructor that sets input and output dimensions
    FlattenLayer(const std::vector<int>& input_shape, const std::vector<int>& output_shape);

    Eigen::MatrixXf forward(const Eigen::MatrixXf& input) override {
        throw std::runtime_error("Flatten Layer does not support forward pass with single input matrix. Please provide a tensor instead.");
    }

    Eigen::MatrixXf backward(const Eigen::MatrixXf& grad) override {
        throw std::runtime_error("Flatten Layer does not support backward pass with single input matrix. Please provide a tensor instead.");
    }

    // Forward pass: Flatten 3D tensor into 2D matrix
    Eigen::MatrixXf forward(const std::vector<Eigen::MatrixXf>& input);

    // Backward pass: Reshape gradients back into 3D tensor
    std::vector<Eigen::MatrixXf> backward(const Eigen::MatrixXf& grad, bool flag);

    // Utility functions
    const std::vector<int>& InputShape() const { return input_shape; }
    const std::vector<int>& OutputShape() const { return output_shape; }
};

#endif //FLATTEN_LAYER_H