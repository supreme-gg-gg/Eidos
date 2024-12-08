#include <gtest/gtest.h>
#include <Eigen/Dense>
#include <algorithm>
#include "../include/activation_fns.h"

// Unit tests for ReLU
TEST(ActivationFunctionsTest, ReLUNegativeInput) {
    Eigen::MatrixXf input(2, 2);
    input << -1, -2, -3, -4;
    Eigen::MatrixXf expected(2, 2);
    expected << 0, 0, 0, 0;
    ReLU relu;
    ASSERT_TRUE(relu.forward(input).isApprox(expected));
}

TEST(ActivationFunctionsTest, ReLUPositiveInput) {
    Eigen::MatrixXf input(2, 2);
    input << 1, 2, 3, 4;
    Eigen::MatrixXf expected(2, 2);
    expected << 1, 2, 3, 4;
    ReLU relu;
    ASSERT_TRUE(relu.forward(input).isApprox(expected));
}

// Unit tests for ReLU backward
TEST(ActivationFunctionsTest, ReLUBackward) {
    Eigen::MatrixXf input(2, 2);
    input << -1, 2, -3, 4;
    Eigen::MatrixXf grad_output(2, 2);
    grad_output << 1, 1, 1, 1;
    Eigen::MatrixXf expected(2, 2);
    expected << 0, 1, 0, 1;
    ReLU relu;
    relu.forward(input);
    ASSERT_TRUE(relu.backward(grad_output).isApprox(expected));
}

// Unit tests for LeakyReLU
TEST(ActivationFunctionsTest, LeakyReLUNegativeInput) {
    Eigen::MatrixXf input(2, 2);
    input << -1, -2, -3, -4;
    Eigen::MatrixXf expected(2, 2);
    expected << -0.01, -0.02, -0.03, -0.04;
    LeakyReLU leaky_relu(0.01);
    ASSERT_TRUE(leaky_relu.forward(input).isApprox(expected, 1e-6));
}

TEST(ActivationFunctionsTest, LeakyReLUPositiveInput) {
    Eigen::MatrixXf input(2, 2);
    input << 1, 2, 3, 4;
    Eigen::MatrixXf expected(2, 2);
    expected << 1, 2, 3, 4;
    LeakyReLU leaky_relu(0.01);
    ASSERT_TRUE(leaky_relu.forward(input).isApprox(expected));
}

// Unit tests for LeakyReLU backward
TEST(ActivationFunctionsTest, LeakyReLUBackward) {
    Eigen::MatrixXf input(2, 2);
    input << -1, 2, -3, 4;
    Eigen::MatrixXf grad_output(2, 2);
    grad_output << 1, 1, 1, 1;
    Eigen::MatrixXf expected(2, 2);
    expected << 0.01, 1, 0.01, 1;
    LeakyReLU leaky_relu(0.01);
    leaky_relu.forward(input);
    ASSERT_TRUE(leaky_relu.backward(grad_output).isApprox(expected));
}

