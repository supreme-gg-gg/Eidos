#include "../include/model.h"
#include "../include/optimizer.h"
#include "../include/loss_fns.h"
#include "../include/dense_layer.h"
#include "../include/activation_fns.h"
#include <Eigen/Dense>
#include <iostream>

int main() {
    // Create a model
    Model model;
    
    // Add layers to the model
    model.Add(new DenseLayer(10, 5));
    model.Add(new ReLU());
    model.Add(new DenseLayer(5, 1));  // Output layer with 1 neuron for regression

    // Provide input and target data
    Eigen::MatrixXf inputs = Eigen::MatrixXf::Random(10, 3);  // 3 samples
    Eigen::MatrixXf targets = Eigen::MatrixXf::Random(1, 3);  // 3 target outputs for regression

    // Set optimizer and loss function
    SGD optimizer(0.01);
    MSELoss loss_fn;
    model.set_optimizer(optimizer);

    // Training loop
    for (int epoch = 0; epoch < 100; ++epoch) {
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