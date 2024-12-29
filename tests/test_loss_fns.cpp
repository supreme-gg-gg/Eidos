#include "../include/Eidos/loss_fns.h"
#include <gtest/gtest.h>
#include <Eigen/Dense>

// Test CrossEntropyLoss
TEST(LossTest, CategoricalCrossEntropyLoss_ComputesCorrectly) {

    CategoricalCrossEntropyLoss loss;
    
    Eigen::MatrixXf predictions(2, 4);
    predictions << 0.25, 0.25, 0.25, 0.25,
                    0.01, 0.01, 0.01, 0.96;

    Eigen::MatrixXf targets(2, 4);
    targets << 0, 0, 0, 1,
                0, 0, 0, 1;

    // Compute the loss for the new test case
    float loss_value = loss.forwardMatrix(predictions, targets);

    // Check if loss is computed correctly for the new test case
    EXPECT_NEAR(loss_value, 0.713558, 1e-5);
}

TEST(LossTest, CrossEntropyLoss_ComputesCorrectly) {

    CrossEntropyLoss loss;

    Eigen::MatrixXf logits(2, 4);
    logits << 1.2, 0.9, 0.5, 0.1,
              2.1, 1.5, 0.3, 0.7;

    Eigen::MatrixXf targets(2, 4);
    targets << 0, 0, 0, 1,
               0, 0, 1, 0;
    
    float loss_value = loss.forwardMatrix(logits, targets);
    EXPECT_NEAR(loss_value, 2.258659, 1e-5);
}

TEST(LossTest, CrossEntropyLoss_Backward) {
    CrossEntropyLoss loss;
    Eigen::MatrixXf logits(2, 4);
    Eigen::MatrixXf targets(2, 4);
    logits << 1.2, 0.9, 0.5, 0.1,
              2.1, 1.5, 0.3, 0.7;
    targets << 0, 0, 0, 1,
               0, 0, 1, 0;

    float computed_loss = loss.forwardMatrix(logits, targets);
    Eigen::MatrixXf grad = loss.backwardMatrix();
    ASSERT_EQ(grad.rows(), 2);
    ASSERT_EQ(grad.cols(), 4);
}

// Test MSELoss
TEST(LossTest, MSELoss_ComputesCorrectly) {
    // Example inputs (predictions and targets)
    Eigen::MatrixXf predictions(3, 2);
    predictions << 0.2, 0.5,
                   0.3, 0.1,
                   0.3, 0.6;

    Eigen::MatrixXf targets(3, 2);
    targets << 0.1, 0.5,
               0.2, 0.0,
               0.3, 0.7;

    // Instantiate the loss function
    MSELoss loss;
    
    // Compute the loss
    float loss_value = loss.forwardMatrix(predictions, targets);
    
    // Check if loss is computed correctly (here you can check against expected value)
    EXPECT_NEAR(loss_value, 0.006667, 1e-5);
}

TEST(LossTest, MSELoss_ForwardAndBackward) {
    MSELoss loss;
    Eigen::MatrixXf predictions(3, 2);
    Eigen::MatrixXf targets(3, 2);
    predictions << 0.5, 0.8,
                   0.2, 0.3,
                   0.1, 0.4;
    targets << 0.0, 1.0,
               0.0, 0.0,
               0.0, 1.0;

    loss.forwardMatrix(predictions, targets);
    Eigen::MatrixXf grad = loss.backwardMatrix();
    ASSERT_EQ(grad.rows(), 3);
    ASSERT_EQ(grad.cols(), 2);
}