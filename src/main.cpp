#include "../include/dense_layer.h"
#include "../include/activation_fns.h"
#include "../include/model.h"
#include "../include/optimizer.h"
#include "../include/layer.h"
#include "../include/mse_loss.h"
#include <Eigen/Dense>
#include <iostream>

int main() {
    // Create a model
    Model model;
    
    // Add layers to the model using simplified methods
    model.add_dense_layer(10, 5);  // Dense layer (10 -> 5)
    model.add_relu_layer();        // ReLU activation
    model.add_dense_layer(5, 2);   // Dense layer (5 -> 2)

    Eigen::MatrixXf inputs = Eigen::MatrixXf::Random(10, 3);  // 3 samples
    Eigen::MatrixXf targets = Eigen::MatrixXf::Random(2, 3); // 3 target outputs

    SGD optimizer(0.01);
    MSELoss loss_fn;

    // Training loop
    for (int epoch = 0; epoch < 100; ++epoch) {
        Eigen::MatrixXf outputs = model.forward(inputs);
        float loss = loss_fn.forward(outputs, targets);
        Eigen::MatrixXf grad_loss = loss_fn.backward();

        model.backward(grad_loss);
        model.optimize(optimizer);

        std::cout << "Epoch " << epoch << " completed. Loss: " << loss << std::endl;
    }

    return 0;
}