#include "../../include/model.h"
#include "../../include/optimizer.h"
#include "../../include/loss_fns.h"
#include "../../include/layers.h"
#include "../../include/activation_fns.h"
#include "../../include/debugger.hpp"
#include "../../include/callback.h"

#include <iostream>
#include <Eigen/Dense>

int main() {
    // Sample data for MLP training
    int num_samples = 100;   // Number of samples
    int input_size = 20;     // Input feature size
    int output_size = 5;     // Number of classes (one-hot encoded targets)

    // Random initialization of inputs and targets
    Eigen::MatrixXf inputs = Eigen::MatrixXf::Random(num_samples, input_size);
    Eigen::MatrixXf one_hot_targets(num_samples, output_size);
    
    // Simulate one-hot encoded targets
    for (int i = 0; i < num_samples; ++i) {
        one_hot_targets.row(i).setZero();
        int target_class = rand() % output_size;
        one_hot_targets(i, target_class) = 1.0;
    }

    // Model setup
    Model model;
    model.Add(new DenseLayer(input_size, 64));
    model.Add(new ReLU());
    model.Add(new DenseLayer(64, 32));
    model.Add(new ReLU());
    model.Add(new DenseLayer(32, output_size));
    Adam optimizer(0.001);
    CrossEntropyLoss loss_fn;
    model.set_optimizer(optimizer);

    Debugger debugger;
    debugger.track_layer(model.get_layer(2));

    model.add_callback(new EarlyStopping(5));
    model.Train(inputs, one_hot_targets, 50, 32, loss_fn);

    // Training loop
    // for (int epoch = 0; epoch < 10; ++epoch) {

    //     debugger.save_previous_weights();

    //     // Forward pass
    //     Eigen::MatrixXf output = model.forward(inputs);

    //     // Compute loss
    //     float loss = loss_fn.forward(output, one_hot_targets);
    //     std::cout << "Epoch: " << epoch << " Loss: " << loss << std::endl;

    //     // Backward pass
    //     Eigen::MatrixXf grad = loss_fn.backward();  // Backprop through loss function
    //     model.backward(grad);                      // Backprop through layers

    //     // Update weights
    //     model.optimize();

    //     debugger.print_weight_change_norms();
    // }

    return 0;
}