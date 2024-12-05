#include "../include/dense_layer.h"
#include "../include/optimizer.h"
#include "../include/mse_loss.h"
#include <iostream>
#include <Eigen/Dense>

int main() {

    int num_epochs = 100; // Define the number of epochs
    DenseLayer layer1(784, 128); // Example: 784 inputs to 128 outputs (hidden layer)
    DenseLayer layer2(128, 10);  // Example: 128 inputs to 10 outputs (output layer)

    MSELoss mse_loss;  // Mean Squared Error loss function

    SGD optimizer(0.01); // Define SGD optimizer with learning rate of 0.01

    // Example data (one image and one target vector)
    Eigen::MatrixXf inputs(3, 784);
    Eigen::MatrixXf targets(3, 10);

    // Initialize inputs and targets for demonstration purposes
    inputs.setRandom();
    targets.setRandom();

    for (int epoch = 0; epoch < num_epochs; ++epoch) {
        // Forward pass: Calculate outputs for both layers
        Eigen::MatrixXf output1 = layer1.forward(inputs);
        Eigen::MatrixXf output2 = layer2.forward(output1);

        // Compute MSE loss (Mean Squared Error)
        float loss = mse_loss.forward(output2, targets);

        // Backward pass: Calculate gradients
        Eigen::MatrixXf output2_grad = mse_loss.backward();
        Eigen::MatrixXf output1_grad = layer2.backward(output2_grad);  // Backward pass for layer2
        Eigen::MatrixXf input_grad = layer1.backward(output1_grad);  // Backward pass for layer1

        // Update weights and biases using SGD optimizer
        layer1.update_weights(optimizer);
        layer2.update_weights(optimizer);

        // Output loss for monitoring progress
        std::cout << "Epoch " << epoch << ", Loss: " << loss << std::endl;
    }

    return 0;
}
