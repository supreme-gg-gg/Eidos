#include "../include/activation_fns.h"
#include <Eigen/Dense>
#include <iostream>

Eigen::MatrixXf ReLU::forward(const Eigen::MatrixXf& input) {
    cache_output = (input.array() > 0).cast<float>(); // Cache the mask
    return input.cwiseMax(0);                        // Apply ReLU activation
}

Tensor<Eigen::MatrixXf> ReLU::forward(const Tensor<Eigen::MatrixXf>& input) {

    Tensor<Eigen::MatrixXf> cache_tensor(input.depth(), input[0].rows(), input[0].cols());

    for (int i = 0; i < input.depth(); ++i) {  // Iterate over the first dimension (batch or depth)

        cache_tensor[i] = this->forward(input[i]);  // Apply ReLU to each matrix (slice of the tensor)
        
        // Cache the output mask if necessary (to be used during backward pass)
        cache_output_tensor.push_back(this->cache());
    }

    return cache_tensor;  // Return the tensor with ReLU applied to each slice
}

Eigen::MatrixXf ReLU::backward(const Eigen::MatrixXf& grad_output) {
    return grad_output.array() * cache_output.array(); // Apply the cached mask to gradients
}

Tensor<Eigen::MatrixXf> ReLU::backward(const Tensor<Eigen::MatrixXf>& grad_output) {
    Tensor<Eigen::MatrixXf> grad_tensor(grad_output.depth(), grad_output[0].rows(), grad_output[0].cols());

    for (int i = 0; i < grad_output.depth(); ++i) {
        grad_tensor[i] = this->backward(grad_output[i]);  // Apply ReLU backward to each matrix
    }

    return grad_tensor;
}

Eigen::MatrixXf LeakyReLU::forward(const Eigen::MatrixXf& input) {
    cache_output = (input.array() > 0).cast<float>() + alpha * (input.array() <= 0).cast<float>();
    return input.cwiseMax(0) + alpha * input.cwiseMin(0); // Leaky ReLU activation
}

Tensor<Eigen::MatrixXf> LeakyReLU::forward(const Tensor<Eigen::MatrixXf>& input) {

    Tensor<Eigen::MatrixXf> cache_tensor(input.depth(), input[0].rows(), input[0].cols());

    for (int i = 0; i < input.depth(); ++i) {
        cache_tensor[i] = this->forward(input[i]);
        cache_output_tensor.push_back(this->cache());
    }

    return cache_tensor;
}

Eigen::MatrixXf LeakyReLU::backward(const Eigen::MatrixXf& grad_output) {
    return grad_output.array() * cache_output.array(); // Apply the cached mask to gradients
}

Tensor<Eigen::MatrixXf> LeakyReLU::backward(const Tensor<Eigen::MatrixXf>& grad_output) {
    Tensor<Eigen::MatrixXf> grad_tensor(grad_output.depth(), grad_output[0].rows(), grad_output[0].cols());

    for (int i = 0; i < grad_output.depth(); ++i) {
        grad_tensor[i] = this->backward(grad_output[i]);
    }

    return grad_tensor;
}

Eigen::MatrixXf Sigmoid::forward(const Eigen::MatrixXf& input) {
    cache_output = 1.0f / (1.0f + (-input.array()).exp());
    return cache_output;
}

Tensor<Eigen::MatrixXf> Sigmoid::forward(const Tensor<Eigen::MatrixXf>& input) {

    Tensor<Eigen::MatrixXf> cache_tensor(input.depth(), input[0].rows(), input[0].cols());

    for (int i = 0; i < input.depth(); ++i) {
        cache_tensor[i] = this->forward(input[i]);
        cache_output_tensor.push_back(this->cache());
    }

    return cache_tensor;
}

Eigen::MatrixXf Sigmoid::backward(const Eigen::MatrixXf& grad_output) {
    Eigen::MatrixXf sigmoid_grad = cache_output.array() * (1.0f - cache_output.array());
return grad_output.array() * sigmoid_grad.array();
}

Tensor<Eigen::MatrixXf> Sigmoid::backward(const Tensor<Eigen::MatrixXf>& grad_output) {
    Tensor<Eigen::MatrixXf> grad_tensor(grad_output.depth(), grad_output[0].rows(), grad_output[0].cols());

    for (int i = 0; i < grad_output.depth(); ++i) {
        grad_tensor[i] = this->backward(grad_output[i]);
    }

    return grad_tensor;
}

Eigen::MatrixXf Softmax::forward(const Eigen::MatrixXf& logits) {
    // Compute the exponentials in a numerically stable way
    Eigen::MatrixXf exp_logits = (logits.array().rowwise() - logits.colwise().maxCoeff().array()).exp();
    Eigen::VectorXf row_sums = exp_logits.array().rowwise().sum();

    float epsilon = 1e-10f;  // Small constant to avoid division by zero
    cache_output = exp_logits.array().colwise() / (row_sums.array() + epsilon);

    return cache_output;
}

Tensor<Eigen::MatrixXf> Softmax::forward(const Tensor<Eigen::MatrixXf>& logits) {

    Tensor<Eigen::MatrixXf> cache_tensor(logits.depth(), logits[0].rows(), logits[0].cols());

    for (int i = 0; i < logits.depth(); ++i) {
        cache_tensor[i] = this->forward(logits[i]);
        cache_output_tensor.push_back(this->cache());
    }

    return cache_tensor;
}

Eigen::MatrixXf Softmax::backward(const Eigen::MatrixXf& grad_output) {
    Eigen::MatrixXf grad = Eigen::MatrixXf::Zero(grad_output.rows(), grad_output.cols());
    for (int i = 0; i < grad.rows(); ++i) {
        Eigen::RowVectorXf y = cache_output.row(i);  // Softmax output for the current sample
        float dot = grad_output.row(i).dot(y);      // Inner product
        grad.row(i) = grad_output.row(i).cwiseProduct(y) - y * dot;
    }
    return grad;
}

Tensor<Eigen::MatrixXf> Softmax::backward(const Tensor<Eigen::MatrixXf>& grad_output) {
    Tensor<Eigen::MatrixXf> grad_tensor(grad_output.depth(), grad_output[0].rows(), grad_output[0].cols());

    for (int i = 0; i < grad_output.depth(); ++i) {
        grad_tensor[i] = this->backward(grad_output[i]);
    }

    return grad_tensor;
}

Eigen::MatrixXf Tanh::forward(const Eigen::MatrixXf& input) {
    cache_output = input.array().tanh();
    return cache_output;
}

Tensor<Eigen::MatrixXf> Tanh::forward(const Tensor<Eigen::MatrixXf>& input) {

    Tensor<Eigen::MatrixXf> cache_tensor(input.depth(), input[0].rows(), input[0].cols());

    for (int i = 0; i < input.depth(); ++i) {
        cache_tensor[i] = this->forward(input[i]);
        cache_output_tensor.push_back(this->cache());
    }

    return cache_tensor;
}

Eigen::MatrixXf Tanh::backward(const Eigen::MatrixXf& grad_output) {
    return grad_output.array() * (1 - cache_output.array().square());
}

Tensor<Eigen::MatrixXf> Tanh::backward(const Tensor<Eigen::MatrixXf>& grad_output) {
    Tensor<Eigen::MatrixXf> grad_tensor(grad_output.depth(), grad_output[0].rows(), grad_output[0].cols());

    for (int i = 0; i < grad_output.depth(); ++i) {
        grad_tensor[i] = this->backward(grad_output[i]);
    }

    return grad_tensor;
}