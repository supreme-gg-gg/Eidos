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
    // Sample data for MLP training with batching
    int num_samples = 100;   // Number of samples
    int input_size = 20;     // Input feature size
    int output_size = 5;     // Number of classes (one-hot encoded targets)
    int batch_size = 10;     // Batch size
    int num_batches = num_samples / batch_size; // Number of batches

    Tensor input_tensor(num_batches, batch_size, input_size);
    Eigen::MatrixXf input_data = Eigen::MatrixXf::Random(num_samples, input_size);

    Tensor one_hot_targets(num_batches, batch_size, output_size);
    Eigen::MatrixXf output_labels = Eigen::MatrixXf::Zero(num_samples, output_size);
    
    // Simulate one-hot encoded targets
    for (int i = 0; i < num_samples; ++i) {
        output_labels.row(i).setZero();
        int target_class = rand() % output_size;
        output_labels(i, target_class) = 1.0;
    }

    // Reshape the tensors to match the batch size
    for (int b = 0; b < num_batches; ++b) {
        input_tensor[b] = input_data.block(b * batch_size, 0, batch_size, input_size);
        one_hot_targets[b] = output_labels.block(b * batch_size, 0, batch_size, output_size);
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

    // Training loop
    // for (int epoch = 0; epoch < 10; ++epoch) {

    //     // Forward pass
    //     Tensor output = model.forward(input_tensor);

    //     // Compute loss
    //     float loss = loss_fn.forward(output, one_hot_targets);
    //     std::cout << "Epoch: " << epoch << " Loss: " << loss << std::endl;

    //     // Backward pass
    //     Tensor grad = loss_fn.backward();  // Backprop through loss function
    //     model.backward(grad);                      // Backprop through layers

    //     // Update weights
    //     model.optimize();
    // }

    // model.add_callback(new EarlyStopping(5));
    model.Train(input_tensor, one_hot_targets, 50, loss_fn);

    // Sample testing data
    Eigen::MatrixXf test_input_data = Eigen::MatrixXf::Random(batch_size, input_size);
    Tensor test_input_tensor = Tensor(test_input_data);

    Eigen::MatrixXf test_output_labels = Eigen::MatrixXf::Zero(batch_size, output_size);
    for (int i = 0; i < batch_size; ++i) {
        test_output_labels.row(i).setZero();
        int target_class = rand() % output_size;
        test_output_labels(i, target_class) = 1.0;
    }
    Tensor test_one_hot_targets = Tensor(test_output_labels);

    // Testing
    model.Test(test_input_tensor, test_one_hot_targets, loss_fn);

    return 0;
}