#include <gtest/gtest.h>
#include <Eigen/Dense>
#include "../include/layers/dense_layer.h"
#include "../include/loss_fns.h"

TEST(DenseLayerTest, ForwardPassCorrectShape) {
    DenseLayer layer(10, 5); // 10 input features, 5 outputs
    Eigen::MatrixXf inputs(3, 10); // 3 samples, 10 features each
    inputs.setRandom();

    Tensor outputs = layer.forward(Tensor(inputs));
    ASSERT_EQ(outputs.getSingleMatrix().rows(), 3);
    ASSERT_EQ(outputs.getSingleMatrix().cols(), 5);
}

TEST(DenseLayerTest, BackwardPassCorrectShapes) {
    DenseLayer layer(10, 5);
    Eigen::MatrixXf inputs(3, 10);
    Eigen::MatrixXf grad_output(3, 5);
    inputs.setRandom();
    grad_output.setRandom();

    layer.forward(Tensor(inputs));
    Tensor grad_input_tensor = layer.backward(Tensor(grad_output));
    Eigen::MatrixXf grad_input = grad_input_tensor.getSingleMatrix();

    // Verify gradient shapes
    ASSERT_EQ(grad_input.rows(), 3); // Same as input features
    ASSERT_EQ(grad_input.cols(), 10);  // Same as number of samples
    ASSERT_EQ(layer.get_grad_weights()[0]->rows(), 3); // Matches output features
    ASSERT_EQ(layer.get_grad_weights()[0]->cols(), 5); // Matches input features
    ASSERT_EQ(layer.get_grad_bias()[0]->size(), 5); // Matches output features
}