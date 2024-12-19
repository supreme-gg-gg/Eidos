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
 * 
 * @param input_shape: The dimensions of the input tensor (e.g., {Channels, Height, Width})
 * @param output_shape: The dimensions of the flattened output (e.g., {Channels, Flattened_Size})
 */
class FlattenLayer : public Layer {
private:
    std::vector<int> input_shape;  // Input dimensions (e.g., {Channels, Height, Width})
    std::vector<int> output_shape; // Flattened dimensions (e.g., {Channels, Flattened_Size})

public:
    /**
     * @brief Constructs a FlattenLayer with the specified input and output shapes.
     * 
     * @param input_shape A constant reference to a vector representing the shape of the input tensor.
     * @param output_shape A constant reference to a vector representing the shape of the output tensor.
     */
    FlattenLayer(const std::vector<int>& input_shape, const std::vector<int>& output_shape);

    /**
     * @brief Applies a forward transformation to flatten the input matrices.
     * 
     * This function takes a vector of Eigen::MatrixXf objects as input and 
     * performs a forward transformation to flatten them into a single 
     * Eigen::MatrixXf object.
     * 
     * @param input A vector of Eigen::MatrixXf objects representing the input matrices.
     * @return Eigen::MatrixXf The resulting flattened matrix.
     */
    Tensor forward(const Tensor& input) override;

    /**
     * @brief Performs the backward transformation for the flatten layer.
     *
     * This function takes the gradient of the loss with respect to the output of the flatten layer
     * and transforms it back to the shape of the input to the flatten layer.
     *
     * @param grad The gradient of the loss with respect to the output of the flatten layer.
     * @return A vector of Eigen::MatrixXf representing the transformed gradients.
     */
    Tensor backward(const Tensor& grad) override;

    /**
     * @brief Get the shape of the input tensor.
     * 
     * This function returns a constant reference to a vector containing the dimensions
     * of the input tensor. The dimensions are stored in the order of the tensor's axes.
     * 
     * @return const std::vector<int>& A constant reference to the vector representing the input shape.
     */
    const std::vector<int>& InputShape() const { return input_shape; }

    /**
     * @brief Get the output shape of the flatten layer.
     * 
     * This function returns a constant reference to the vector representing
     * the shape of the output tensor after the flatten operation.
     * 
     * @return const std::vector<int>& A constant reference to the output shape vector.
     */
    const std::vector<int>& OutputShape() const { return output_shape; }
};

#endif //FLATTEN_LAYER_H