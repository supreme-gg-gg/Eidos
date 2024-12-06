#include "../include/cross_entropy_loss.h"
#include "../include/mse_loss.h"
#include <gtest/gtest.h>
#include <Eigen/Dense>

// Test CrossEntropyLoss
TEST(LossTest, CrossEntropyLoss_ComputesCorrectly) {
    // Example inputs (predictions and targets)
    Eigen::MatrixXf predictions(2, 3);
    predictions << 0.2, 0.5, 0.3,  // Prediction for class 1
                   0.1, 0.3, 0.6;  // Prediction for class 2

    Eigen::MatrixXf targets(2, 3);
    targets << 0, 1, 0,  // Target is class 2
               0, 0, 1;  // Target is class 3

    // Instantiate the loss function
    CrossEntropyLoss loss;
    
    // Compute the loss
    float loss_value = loss.forward(predictions, targets);
    
    // Check if loss is computed correctly (here you can check against expected value)
    EXPECT_FLOAT_EQ(loss_value, 1.204);
}

// Test MSELoss
TEST(LossTest, MSELoss_ComputesCorrectly) {
    // Example inputs (predictions and targets)
    Eigen::MatrixXf predictions(2, 3);
    predictions << 0.2, 0.5, 0.3,
                   0.1, 0.3, 0.6;

    Eigen::MatrixXf targets(2, 3);
    targets << 0.1, 0.5, 0.2,
               0.0, 0.3, 0.7;

    // Instantiate the loss function
    MSELoss loss;
    
    // Compute the loss
    float loss_value = loss.forward(predictions, targets);
    
    // Check if loss is computed correctly (here you can check against expected value)
    EXPECT_FLOAT_EQ(loss_value, 0.021);
}
