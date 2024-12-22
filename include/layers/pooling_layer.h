#ifndef POOLING_LAYER_H
#define POOLING_LAYER_H

#include <Eigen/Dense>
#include "../layer.h"

class MaxPooling2D: public Layer {
public:
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

class AveragePooling2D: public Layer {
public:
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