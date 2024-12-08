#include <gtest/gtest.h>
#include <Eigen/Dense>
#include "../include/activation_fns.h"
#include "../include/loss_fns.h"

// Test Sigmoid forward pass
TEST(SigmoidTest, ForwardPass) {
    Sigmoid sigmoid;

    Eigen::MatrixXf inputs(5, 1);
    inputs << 0.1f, -0.5f, 2.0f, -1.0f, 3.0f;

    Eigen::MatrixXf outputs = sigmoid.forward(inputs);

    // Check that the output is between 0 and 1 (Sigmoid output range)
    ASSERT_TRUE((outputs.array() >= 0.0f).all());
    ASSERT_TRUE((outputs.array() <= 1.0f).all());
}

// Test Binary Cross-Entropy Loss forward
TEST(BinaryCrossEntropyLossTest, LossForward) {
    BinaryCrossEntropyLoss loss_fn;

    Eigen::MatrixXf predictions(5, 1);
    predictions << 0.9f, 0.1f, 0.8f, 0.2f, 0.7f;  // Predicted values (sigmoid outputs)

    Eigen::MatrixXf targets(5, 1);
    targets << 1.0f, 0.0f, 1.0f, 0.0f, 1.0f;  // Binary targets

    float loss = loss_fn.forward(predictions, targets);

    // Assert that the loss is a non-negative value
    ASSERT_GT(loss, 0);
}

// Test Binary Cross-Entropy Loss backward
TEST(BinaryCrossEntropyLossTest, LossBackward) {
    BinaryCrossEntropyLoss loss_fn;

    Eigen::MatrixXf predictions(5, 1);
    predictions << 0.9f, 0.1f, 0.8f, 0.2f, 0.7f;  // Predicted values (sigmoid outputs)

    Eigen::MatrixXf targets(5, 1);
    targets << 1.0f, 0.0f, 1.0f, 0.0f, 1.0f;  // Binary targets

    loss_fn.forward(predictions, targets);  // Compute the forward pass to set predictions

    Eigen::MatrixXf grad_loss = loss_fn.backward();

    // Ensure gradients have the correct shape (same as the predictions)
    ASSERT_EQ(grad_loss.rows(), 5);
    ASSERT_EQ(grad_loss.cols(), 1);
    ASSERT_FALSE(grad_loss.isZero());
}