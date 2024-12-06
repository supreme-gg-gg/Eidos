#include <gtest/gtest.h>
#include <Eigen/Dense>
#include "../include/dense_layer.h"
#include "../include/mse_loss.h"

TEST(DenseLayerTest, ForwardPassCorrectShape) {
    DenseLayer layer(10, 5); // 10 input features, 5 outputs
    Eigen::MatrixXf inputs(10, 3); // 3 samples, 10 features each
    inputs.setRandom();

    Eigen::MatrixXf outputs = layer.forward(inputs);
    ASSERT_EQ(outputs.rows(), 5); // Output should have 5 rows (output features)
    ASSERT_EQ(outputs.cols(), 3); // Output should have 3 columns (samples)
}

TEST(DenseLayerTest, BackwardPassCorrectShapes) {
    DenseLayer layer(10, 5);
    Eigen::MatrixXf inputs(10, 3);
    Eigen::MatrixXf grad_output(5, 3); // Same shape as forward pass output
    inputs.setRandom();
    grad_output.setRandom();

    layer.forward(inputs);
    Eigen::MatrixXf grad_input = layer.backward(grad_output);

    // Verify gradient shapes
    ASSERT_EQ(grad_input.rows(), 10); // Same as input features
    ASSERT_EQ(grad_input.cols(), 3);  // Same as number of samples
    ASSERT_EQ(layer.get_weights_gradient().rows(), 5); // Matches output features
    ASSERT_EQ(layer.get_weights_gradient().cols(), 10); // Matches input features
    ASSERT_EQ(layer.get_bias_gradient().size(), 5); // Matches output features
}

TEST(MSELossTest, ForwardAndBackward) {
    MSELoss loss;
    Eigen::MatrixXf predictions(2, 3); // 2 outputs, 3 samples
    Eigen::MatrixXf targets(2, 3);
    predictions << 0.5, 0.8, 0.2,
                   0.3, 0.1, 0.4;
    targets << 0.0, 1.0, 0.0,
               0.0, 0.0, 1.0;

    float computed_loss = loss.forward(predictions, targets);
    ASSERT_NEAR(computed_loss, 0.2133, 1e-4); // Validate expected loss value

    Eigen::MatrixXf grad = loss.backward();
    ASSERT_EQ(grad.rows(), 2); // Should match predictions
    ASSERT_EQ(grad.cols(), 3); // Should match samples
}