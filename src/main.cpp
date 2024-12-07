#include "../include/model.h"
#include "../include/optimizer.h"
#include "../include/loss_fns.h"
#include "../include/dense_layer.h"
#include "../include/activation_fns.h"
#include <Eigen/Dense>
#include <iostream>

int main() {

    // TODO: train on MNIST

    // Create a model
    Model model;
    
    // Add layers to the model
    model.Add(new DenseLayer(64, 32));
    model.Add(new ReLU());
    model.Add(new DenseLayer(32, 16));
    model.Add(new ReLU());
    model.Add(new DenseLayer(16, 10));  // Output layer with 10 neurons for multiclass classification
    model.Add(new Softmax());  // Softmax activation for multiclass classification

    // Provide input and target data
    Eigen::MatrixXf inputs = Eigen::MatrixXf::Random(64, 5);  // 5 samples with 64 features each
    Eigen::MatrixXf targets = Eigen::MatrixXf::Random(10, 5).unaryExpr([](float elem) { return elem > 0.5 ? 1.0f : 0.0f; });  // 5 samples with 10 classes each

    // Set optimizer and loss function
    SGD optimizer(0.01);
    CategoricalCrossEntropyLoss loss_fn;
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