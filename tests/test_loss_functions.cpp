#include "../include/loss_fns.h"
#include <gtest/gtest.h>
#include <Eigen/Dense>

// Test CrossEntropyLoss
TEST(LossTest, CrossEntropyLoss_ComputesCorrectly) {
    // Example inputs (predictions and targets)
    Eigen::MatrixXf predictions(3, 2);
    predictions << 0.2, 0.5,  // Prediction for class 1
                   0.1, 0.3,  // Prediction for class 2
                   0.4, 0.4;  // Prediction for class 3

    Eigen::MatrixXf targets(3, 2);
    targets << 0, 1,  // Target is class 2
               1, 0,  // Target is class 1
               0, 1;  // Target is class 2

    // Instantiate the loss function
    CategoricalCrossEntropyLoss loss;
    
    // Compute the loss
    float loss_value = loss.forward(predictions, targets);
    
    // Check if loss is computed correctly (here you can check against expected value)
    EXPECT_FLOAT_EQ(loss_value, 1.497);
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
    float loss_value = loss.forward(predictions, targets);
    
    // Check if loss is computed correctly (here you can check against expected value)
    EXPECT_FLOAT_EQ(loss_value, 0.006667);
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

    float computed_loss = loss.forward(predictions, targets);
    ASSERT_NEAR(computed_loss, 0.1317, 1e-4); // Validate expected loss value

    Eigen::MatrixXf grad = loss.backward();
    ASSERT_EQ(grad.rows(), 3);
    ASSERT_EQ(grad.cols(), 2);
}