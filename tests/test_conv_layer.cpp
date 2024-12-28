#include <gtest/gtest.h>
#include <Eigen/Dense>
#include "../include/Eidos/tensor.hpp"
#include "../include/Eidos/layers/conv_layer.h"
#include "../include/Eidos/layers/pooling_layer.h"

TEST(ConvLayerTest, ForwardPassCorrectShape) {
    Conv2D conv1(3, 16, 3, 1, 1); // 3 input channels, 16 output channels, 3x3 kernel, stride 1, padding 1
    Tensor input_tensor(3, 32, 32); // Example dimensions for CNN input (batch size, channels, height, width)
    input_tensor.set_random();

    Tensor output_tensor = conv1.forward(input_tensor);
    std::tuple output_shape = output_tensor.shape();

    // Calculate the expected output dimensions using the formula:
    // output_height = (input_height - kernel_size + 2 * padding) / stride + 1
    // output_width = (input_width - kernel_size + 2 * padding) / stride + 1
    ASSERT_EQ(std::get<0>(output_shape), 16);
    ASSERT_EQ(std::get<1>(output_shape), 32);
    ASSERT_EQ(std::get<2>(output_shape), 32);
}

TEST(ConvLayerTest, BackwardPassCorrectShape) {
    Conv2D conv1(3, 16, 3, 1, 1); // 3 input channels, 16 output channels, 3x3 kernel, stride 1, padding 1
    Tensor input_tensor(3, 32, 32); // Example dimensions for CNN input (batch size, channels, height, width)
    Tensor grad_output(16, 32, 32); // Gradient of the loss with respect to the output tensor
    input_tensor.set_random();
    grad_output.set_random();

    conv1.forward(input_tensor);
    Tensor grad_input = conv1.backward(grad_output);
    std::tuple grad_input_shape = grad_input.shape();

    ASSERT_EQ(std::get<0>(grad_input_shape), 3);
    ASSERT_EQ(std::get<1>(grad_input_shape), 32);
    ASSERT_EQ(std::get<2>(grad_input_shape), 32);
}
TEST(PoolingLayerTest, ForwardPassCorrectShape) {
    MaxPooling2D pool1(2, 2); // 2x2 pooling kernel, stride 2
    Tensor input_tensor(3, 32, 32); // Example dimensions for CNN input (channels, height, width)
    input_tensor.set_random();

    Tensor output_tensor = pool1.forward(input_tensor);
    std::tuple output_shape = output_tensor.shape();

    // Calculate the expected output dimensions using the formula:
    // output_height = (input_height - pool_size) / stride + 1
    // output_width = (input_width - pool_size) / stride + 1
    ASSERT_EQ(std::get<0>(output_shape), 3);
    ASSERT_EQ(std::get<1>(output_shape), 16);
    ASSERT_EQ(std::get<2>(output_shape), 16);
}

TEST(PoolingLayerTest, BackwardPassCorrectShape) {
    MaxPooling2D pool1(2, 2); // 2x2 pooling kernel, stride 2
    Tensor input_tensor(3, 32, 32); // Example dimensions for CNN input (channels, height, width)
    Tensor grad_output(3, 16, 16); // Gradient of the loss with respect to the output tensor
    input_tensor.set_random();
    grad_output.set_random();

    pool1.forward(input_tensor);
    Tensor grad_input = pool1.backward(grad_output);
    std::tuple grad_input_shape = grad_input.shape();

    ASSERT_EQ(std::get<0>(grad_input_shape), 3);
    ASSERT_EQ(std::get<1>(grad_input_shape), 32);
    ASSERT_EQ(std::get<2>(grad_input_shape), 32);
}

TEST(PoolingLayerTest, AveragePoolingForwardPassCorrectShape) {
    AveragePooling2D pool1(2, 2); // 2x2 pooling kernel, stride 2
    Tensor input_tensor(3, 32, 32); // Example dimensions for CNN input (channels, height, width)
    input_tensor.set_random();

    Tensor output_tensor = pool1.forward(input_tensor);
    std::tuple output_shape = output_tensor.shape();

    // Calculate the expected output dimensions using the formula:
    // output_height = (input_height - pool_size) / stride + 1
    // output_width = (input_width - pool_size) / stride + 1
    ASSERT_EQ(std::get<0>(output_shape), 3);
    ASSERT_EQ(std::get<1>(output_shape), 16);
    ASSERT_EQ(std::get<2>(output_shape), 16);
}

TEST(PoolingLayerTest, AveragePoolingBackwardPassCorrectShape) {
    AveragePooling2D pool1(2, 2); // 2x2 pooling kernel, stride 2
    Tensor input_tensor(3, 32, 32); // Example dimensions for CNN input (channels, height, width)
    Tensor grad_output(3, 16, 16); // Gradient of the loss with respect to the output tensor
    input_tensor.set_random();
    grad_output.set_random();

    pool1.forward(input_tensor);
    Tensor grad_input = pool1.backward(grad_output);
    std::tuple grad_input_shape = grad_input.shape();

    ASSERT_EQ(std::get<0>(grad_input_shape), 3);
    ASSERT_EQ(std::get<1>(grad_input_shape), 32);
    ASSERT_EQ(std::get<2>(grad_input_shape), 32);
}
