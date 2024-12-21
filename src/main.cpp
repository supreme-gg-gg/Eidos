#include "../include/model.h"
#include "../include/optimizer.h"
#include "../include/loss_fns.h"
#include "../include/layers.h"
#include "../include/activation_fns.h"
#include "../include/generic_data_loader.h"
#include "../include/callback.h"
#include "../include/debugger.hpp"
#include <Eigen/Dense>
#include <iostream>

int main() {

    Tensor input_tensor = Tensor(3, 32, 32); // Example dimensions for CNN input (batch size, channels, height, width)

    Conv2D conv1(3, 16, 3, 1, 1); // 3 input channels, 16 output channels, 3x3 kernel, stride 1, padding 1

    // Forward pass
    Tensor output_tensor = conv1.forward(input_tensor);

    // Print the shape of the output tensor
    auto [depth, rows, cols] = output_tensor.shape();
    std::cout << "Output shape: (" << depth << ", " << rows << ", " << cols << ")" << std::endl;

    // Calculate the expected output dimensions using the formula:
    // output_height = (input_height - kernel_size + 2 * padding) / stride + 1
    // output_width = (input_width - kernel_size + 2 * padding) / stride + 1

    int input_height = 32;
    int input_width = 32;
    int kernel_size = 3;
    int stride = 1;
    int padding = 1;

    int output_height = (input_height - kernel_size + 2 * padding) / stride + 1;
    int output_width = (input_width - kernel_size + 2 * padding) / stride + 1;

    std::cout << "Calculated output shape: (" << 16 << ", " << output_height << ", " << output_width << ")" << std::endl;

    // Backward pass
    Tensor grad_output = Tensor(16, output_height, output_width); // Gradient of the loss with respect to the output tensor
    
    // Calculate the gradient of the loss with respect to the input tensor
    Tensor grad_input = conv1.backward(grad_output);

    // Print the shape of the gradient of the input tensor
    auto [grad_depth, grad_rows, grad_cols] = grad_input.shape();
    std::cout << "Gradient of input shape: (" << grad_depth << ", " << grad_rows << ", " << grad_cols << ")" << std::endl;

    std::cout << "Correct output shape: (16, 32, 32)" << std::endl;

    return 0;
}