#ifndef DENSE_LAYER_H
#define DENSE_LAYER_H

#include "../layer.h"
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

    Eigen::MatrixXf forward(const Eigen::MatrixXf& input) override;
    Eigen::MatrixXf backward(const Eigen::MatrixXf& grad_output) override;

    bool has_weights() const override;
    bool has_bias() const override;

    std::vector<Eigen::MatrixXf*> get_weights() override;

    std::vector<Eigen::MatrixXf*> get_grad_weights() override;

    std::vector<Eigen::VectorXf*> get_bias() override;

    std::vector<Eigen::VectorXf*> get_grad_bias() override;
};

#endif //DENSE_LAYER_H