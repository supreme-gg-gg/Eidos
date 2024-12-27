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
    // Input shape of the layer
    std::vector<int> input_shape;

    // Output shape of the layer
    std::vector<int> output_shape;

public:

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

    bool has_weights() const override { return false; }
    bool has_bias() const override { return false; }

    std::string get_name() const override { return "Flatten"; }
    std::string get_details() const override {
        return "   Input Shape: " + std::to_string(input_shape[0]) + "x" + std::to_string(input_shape[1]) + "x" + std::to_string(input_shape[2]) +
            "\n   Output Shape: " + std::to_string(output_shape[0]) + "x" + std::to_string(output_shape[1]) + "\n";
    }
    
    void serialize(std::ofstream& toFileStream) const override {};
    static FlattenLayer* deserialize(std::ifstream& fromFileStream);
};

#endif //FLATTEN_LAYER_H