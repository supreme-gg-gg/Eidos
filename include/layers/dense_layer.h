#ifndef DENSE_LAYER_H
#define DENSE_LAYER_H

#include "../layer.h"
#include "../tensor.hpp"
#include "../optimizer.h"
#include <Eigen/Dense>

/**
 * @class DenseLayer
 * 
 * 此層為重重疊疊之結構，猶如山川重峦，層層疊起，
 * 設計精巧，層層遞進，每一層皆重若千鈞。
 * 
 * 其內部運算，猶如群山之間，流轉無窮，權重與偏置交織，
 * 轉換輸入，成就最終之輸出。
 */
class DenseLayer : public Layer {
private:
    Eigen::MatrixXf weights;
    Eigen::VectorXf bias;
    Eigen::MatrixXf grad_weights;
    Eigen::VectorXf grad_bias;
    Eigen::MatrixXf input;

public:
    DenseLayer(int input_size, int output_size);

    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& grad_output) override;

    bool has_weights() const override;
    bool has_bias() const override;

    std::vector<Eigen::MatrixXf*> get_weights() override;

    std::vector<Eigen::MatrixXf*> get_grad_weights() override;

    std::vector<Eigen::VectorXf*> get_bias() override;

    std::vector<Eigen::VectorXf*> get_grad_bias() override;

    std::string get_name() const override { return "Dense"; }

    std::string get_details() const override {
        return "Input Size: " + std::to_string(weights.rows()) + "\n" +
               "Output Size: " + std::to_string(weights.cols());
    }
    
    /**
     * @brief Serializes the DenseLayer object to a binary stream.
     *
     * This function writes the DenseLayer's weights and biases to the provided output stream in binary format.
     * The structure of the binary serialization is as follows:
     * 
     * 1. Eigen::Index w_rows: Number of rows in the weights matrix.
     * 2. Eigen::Index w_cols: Number of columns in the weights matrix.
     * 3. Eigen::Index b_rows: Number of rows in the bias vector.
     * 4. Eigen::Index b_cols: Number of columns in the bias vector.
     * 5. float[] weights: The weights matrix data, stored in row-major order.
     * 6. float[] bias: The bias vector data.
     * 
     * The sizes of the weights and bias arrays are determined by the dimensions specified in the first four fields.
     * 
     * @param toFileStream The output stream to which the DenseLayer object will be serialized.
     */
    void serialize(std::ofstream& toFileStream) const override;

    static DenseLayer* deserialize(std::ifstream& fromFileName);
};

#endif //DENSE_LAYER_H