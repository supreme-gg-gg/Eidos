#include "../include/model.h"
#include "../include/optimizer.h"
#include "../include/loss_fns.h"
#include "../include/dense_layer.h"
#include "../include/activation_fns.h"
#include <Eigen/Dense>
#include <iostream>

int main() {
    
    Eigen::MatrixXf inputs = Eigen::MatrixXf::Random(5, 64);
    Eigen::MatrixXf targets = Eigen::MatrixXf::Zero(5, 10);
    for (int i = 0; i < targets.rows(); ++i) {
        int index = rand() % targets.cols();
        targets(i, index) = 1.0;
    }

    // Create and set up the model
    Model model;
    model.Add(new DenseLayer(64, 32));
    model.Add(new LeakyReLU());
    model.Add(new DenseLayer(32, 16));
    model.Add(new LeakyReLU());
    model.Add(new DenseLayer(16, 10));  // Output layer with 1 neuron for regression

    // Set optimizer and loss function
    SGD optimizer(0.01);
    CrossEntropyLoss loss_fn;
    model.set_optimizer(optimizer);

    // Training loop
    for (int epoch = 0; epoch < 30; ++epoch) {
        // Forward pass
        Eigen::MatrixXf outputs = model.forward(inputs);

        float loss = loss_fn.forward(outputs, targets);

        // Backward pass
        Eigen::MatrixXf grad_loss = loss_fn.backward();

        model.backward(grad_loss);

        // Optimize
        model.optimize();

        // Print loss
        std::cout << "Epoch " << epoch << " completed. Loss: " << loss << std::endl;
    }

    return 0;
}