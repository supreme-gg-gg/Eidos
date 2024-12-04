#include "../include/cross_entropy_loss.h"
#include "../include/mse_loss.h"
#include "test_loss.h"
#include <iostream>
#include <Eigen/Dense>
#include <cassert>

void test_cross_entropy_loss() {
    CrossEntropyLoss cross_entropy_loss;

    Eigen::MatrixXf predictions(3, 2);
    predictions << 0.9, 0.1,
                   0.2, 0.8,
                   0.3, 0.7;

    Eigen::MatrixXf targets(3, 2);
    targets << 1, 0,
               0, 1,
               0, 1;

    float expected_loss = - (std::log(0.9) + std::log(0.8) + std::log(0.7)) / 3;
    float computed_loss = cross_entropy_loss.compute_loss(predictions, targets);

    assert(std::abs(computed_loss - expected_loss) < 1e-5);
    std::cout << "Cross-entropy loss test passed!" << std::endl;
}

void test_mse_loss() {
    MSELoss mse_loss;

    Eigen::MatrixXf predictions(3, 2);
    predictions << 0.9, 0.1,
                   0.2, 0.8,
                   0.3, 0.7;

    Eigen::MatrixXf targets(3, 2);
    targets << 1, 0,
               0, 1,
               0, 1;

    float expected_loss = (std::pow(0.9 - 1, 2) + std::pow(0.1 - 0, 2) +
                           std::pow(0.2 - 0, 2) + std::pow(0.8 - 1, 2) +
                           std::pow(0.3 - 0, 2) + std::pow(0.7 - 1, 2)) / 6;
    float computed_loss = mse_loss.compute_loss(predictions, targets);

    assert(std::abs(computed_loss - expected_loss) < 1e-5);
    std::cout << "MSE loss test passed!" << std::endl;
}