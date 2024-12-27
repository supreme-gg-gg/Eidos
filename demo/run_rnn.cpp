#include "../../include/model.h"
#include "../../include/optimizer.h"
#include "../../include/loss_fns.h"
#include "../../include/layers.h"
#include "../../include/activation_fns.h"
#include "../../include/debugger.hpp"

#include <iostream>
#include <Eigen/Dense>

int main() {
    // Sample data for Many-to-Many sequence classification (no batching)
    // Input dimensions: sequence_length = 10, input_size = 20
    Eigen::MatrixXf inputs(20, 30);  // 10 time steps, 20 features per time step
    Eigen::MatrixXf targets(20, 10);  // Target: 10 time steps, 5 classes per time step (one-hot)

    // Simulate target classes for each time step and convert to one-hot encoding
    Eigen::MatrixXf one_hot_targets(20, 10);
    for (int t = 0; t < 10; ++t) {
        int target_class = rand() % 10;  // Random class for each time step
        one_hot_targets.setZero();      // Reset the one-hot vector
        one_hot_targets(t, target_class) = 1.0;  // Set the correct class
    }

    // Convert Eigen matrices to Tensor objects
    // You can create batches of inputs and targets for training
    Tensor input_tensor = Tensor(inputs);
    Tensor one_hot_targets_tensor = Tensor(one_hot_targets);

    Model model;
    model.Add(new RNNLayer(30, 64, 10, new Sigmoid(), true));  // RNN layer: input size = 30, hidden size = 64, output size = 10

    Adam optimizer(0.001);
    CrossEntropyLoss loss_fn;

    model.set_optimizer(optimizer);
    
    Debugger debugger;
    debugger.track_layer(model.get_layer(0));

    // Debugging Variables
    Eigen::MatrixXf output;
    float loss;

    for (int epoch = 0; epoch < 40; ++epoch) {

        debugger.save_previous_weights();

        // Forward pass
        Tensor output = model.forward(input_tensor);

        // Compute loss
        float loss = loss_fn.forward(output, one_hot_targets_tensor);
        std::cout << "Epoch: " << epoch << " Loss: " << loss << std::endl;

        // Backward pass
        Tensor grad = loss_fn.backward();  // Backprop through loss function
        model.backward(grad);                      // Backprop through the model layers

        // Update weights
        model.optimize();

        debugger.print_weight_change_norms();
    }

    return 0;
}