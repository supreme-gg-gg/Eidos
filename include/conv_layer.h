#ifndef CONV_LAYER_H
#define CONV_LAYER_H

#include "layer.h"
#include "tensor.h"
#include <Eigen/Dense>

class ConvLayer: public Layer<Tensor<Eigen::MatrixXf>> {
private:
    Tensor<Eigen::MatrixXf> weights;
    Tensor<Eigen::MatrixXf> grad_kernels;
    Eigen::MatrixXf bias;
    Eigen::MatrixXf grad_bias;

    Tensor<Eigen::MatrixXf> input;
    Tensor<Eigen::MatrixXf> output;

public:
    ConvLayer(int in_channels, int out_channels, int kernel_size, int stride, int padding);

    Tensor<Eigen::MatrixXf> forward(const Tensor<Eigen::MatrixXf>& input) override;

    Tensor<Eigen::MatrixXf> backward(const Tensor<Eigen::MatrixXf>& grad_output) override;

    bool has_weights() const override {
        return true;
    }

    bool has_bias() const override {
        return true;
    }

    Tensor<Eigen::MatrixXf>* get_weights() override {
        return &weights;
    }

    Tensor<Eigen::MatrixXf>* get_grad_weights() override {
        return &grad_kernels;
    }
};

#endif //CONV_LAYER_H