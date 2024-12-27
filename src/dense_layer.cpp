#include "../include/layers/dense_layer.h"

DenseLayer::DenseLayer(int input_size, int output_size)
    : weights(Eigen::MatrixXf::Random(input_size, output_size) * std::sqrt(2.0f / (input_size + output_size))),
      bias(Eigen::VectorXf::Zero(output_size)),
      grad_weights(Eigen::MatrixXf::Zero(input_size, output_size)),
      grad_bias(Eigen::VectorXf::Zero(output_size)) {}

Tensor DenseLayer::forward(const Tensor& input) {
    this->input = input.getSingleMatrix();
    return Tensor((this->input * weights).rowwise() + bias.transpose()); // Row wise bias addition
}

Tensor DenseLayer::backward(const Tensor& grad_output) {
    Eigen::MatrixXf grad_output_mat = grad_output.getSingleMatrix();
    grad_weights = input.transpose() * grad_output_mat; // dL/dW = X^T * dL/dY
    grad_bias = grad_output_mat.colwise().sum(); // dL/db = sum(dL/dY)
    return Tensor(grad_output_mat * weights.transpose()); // dL/dX = dL/dY * W^T
}

bool DenseLayer::has_weights() const { return true; }

bool DenseLayer::has_bias() const { return true; }

std::vector<Eigen::MatrixXf*> DenseLayer::get_weights() {
    return { &weights };
}

std::vector<Eigen::MatrixXf*> DenseLayer::get_grad_weights() {
    return { &grad_weights };
}

std::vector<Eigen::VectorXf*> DenseLayer::get_bias() {
    return { &bias };
}

std::vector<Eigen::VectorXf*> DenseLayer::get_grad_bias() {
    return { &grad_bias };
}

void DenseLayer::serialize(std::ofstream& toFileStream) const {
    Eigen::Index w_rows = weights.rows();
    Eigen::Index w_cols = weights.cols();
    Eigen::Index b_rows = bias.rows();
    Eigen::Index b_cols = bias.cols();
    toFileStream.write((char*)&w_rows, sizeof(Eigen::Index));
    toFileStream.write((char*)&w_cols, sizeof(Eigen::Index));
    toFileStream.write((char*)&b_rows, sizeof(Eigen::Index));
    toFileStream.write((char*)&b_cols, sizeof(Eigen::Index));
    toFileStream.write((char*)weights.data(), weights.size() * sizeof(float));
    toFileStream.write((char*)bias.data(), bias.size() * sizeof(float));
}

DenseLayer* DenseLayer::deserialize(std::ifstream& fromFileStream) {
    Eigen::Index w_rows, w_cols, b_rows, b_cols;
    fromFileStream.read((char*)&w_rows, sizeof(Eigen::Index));
    fromFileStream.read((char*)&w_cols, sizeof(Eigen::Index));
    fromFileStream.read((char*)&b_rows, sizeof(Eigen::Index));
    fromFileStream.read((char*)&b_cols, sizeof(Eigen::Index));
    
    DenseLayer* layer = new DenseLayer(w_rows, w_cols);
    fromFileStream.read((char*)layer->get_weights()[0]->data(), w_rows * w_cols * sizeof(float));
    fromFileStream.read((char*)layer->get_bias()[0]->data(), b_rows * b_cols * sizeof(float));
    return layer;
}