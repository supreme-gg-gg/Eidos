#include "../include/model.h"
#include "../include/optimizer.h"
#include "../include/loss_fns.h"
#include "../include/layers.h"
#include "../include/activation_fns.h"
#include "../include/generic_data_loader.h"
#include "../include/callback.h"
#include "../include/debugger.hpp"
#include <Eigen/Dense>
#include <iostream>

// int main() {

//     Timer timer;

//     std::string csvPath;
//     std::cout << "Enter the path to the CSV file: ";
//     std::getline(std::cin, csvPath);
//     GenericDataLoader loader;
//     std::vector<Eigen::MatrixXf> features;
//     std::vector<std::string> labels;
//     loader.load_data(csvPath, features, labels);

//     std::vector<Eigen::MatrixXf> train_features, test_features;
//     std::vector<std::string> train_string_labels, test_string_labels;
//     std::vector<Eigen::MatrixXf> train_labels, test_labels;

//     loader.split_data(features, labels, train_features, train_string_labels, test_features, test_string_labels, 0.8);

//     std::map<std::string, int> label_to_index = {{"Iris-setosa", 0}, {"Iris-versicolor", 1}, {"Iris-virginica", 2}};
//     // std::map<std::string, int> label_to_index = {{"0", 0}, {"1", 1}, {"2", 2}, {"3", 3}, {"4", 4}, {"5", 5}, {"6", 6}, {"7", 7}, {"8", 8}, {"9", 9}};
//     loader.convert_to_one_hot(train_string_labels, train_labels, label_to_index);
//     loader.convert_to_one_hot(test_string_labels, test_labels, label_to_index);

//     // Convert vectors to Eigen matrices for easier manipulation
//     Eigen::MatrixXf inputs(train_features.size(), train_features[0].size());
//     Eigen::MatrixXf targets(train_labels.size(), train_labels[0].size());

//     for (int i = 0; i < train_features.size(); ++i) {
//         inputs.row(i) = train_features[i];
//         targets.row(i) = train_labels[i];
//     }

//     Eigen::MatrixXf test_inputs(test_features.size(), test_features[0].size());
//     Eigen::MatrixXf test_targets(test_labels.size(), test_labels[0].size());

//     for (int i = 0; i < test_features.size(); ++i) {
//         // Also normalize the input and target values
//         test_inputs.row(i) = test_features[i];
//         test_targets.row(i) = test_labels[i];
//     }

//     std::cout << "First 5 rows of targets:" << std::endl;
//     for (int i = 0; i < 5 && i < targets.rows(); ++i) {
//         std::cout << targets.row(i) << std::endl;
//     }

//     // Create and set up the model
//     Model model;
//     model.Add(new DenseLayer(4, 32));
//     model.Add(new BatchNorm(32));
//     model.Add(new LeakyReLU());
//     model.Add(new Dropout(0.2));
//     model.Add(new DenseLayer(32, 16));
//     model.Add(new BatchNorm(16));
//     model.Add(new LeakyReLU());
//     model.Add(new DenseLayer(16, 3)); 

//     // Set optimizer and loss function
//     Adam optimizer(0.01);
//     CrossEntropyLoss loss_fn;

//     // Training
//     model.Train(inputs, targets, 20, 32, loss_fn, &optimizer);

//     // model.set_optimizer(optimizer);
//     // for (int epoch = 0; epoch < 100; ++epoch) {
//     //     Eigen::MatrixXf outputs = model.forward(inputs);
//     //     float loss = loss_fn.forward(outputs, targets);
//     //     Eigen::MatrixXf grad_loss = loss_fn.backward();
//     //     model.backward(grad_loss);
//     //     model.optimize();

//     //     std::cout << "Epoch " << epoch << " completed. Loss: " << loss << std::endl;
//     // }

//     // Testing
//     model.Test(test_inputs, test_targets, loss_fn);

//     return 0;
// }

int main() {
    // Sample data for Many-to-Many sequence classification (no batching)
    int sequence_length = 10;   // Number of timesteps
    int input_size = 50;        // Input feature size per timestep
    int output_size = 10;       // Number of classes (one-hot encoded targets)
    int hidden_size = 64;       // GRU hidden size

    // Random initialization of inputs and targets
    Eigen::MatrixXf inputs = Eigen::MatrixXf::Random(sequence_length, input_size);
    Eigen::MatrixXf one_hot_targets(sequence_length, output_size);
    
    // Simulate one-hot encoded targets
    for (int t = 0; t < sequence_length; ++t) {
        one_hot_targets.row(t).setZero();
        int target_class = rand() % output_size;
        one_hot_targets(t, target_class) = 1.0;
    }

    // Model setup
    Model model;
    model.Add(new GRULayer(input_size, hidden_size, output_size, new Sigmoid(), new Tanh(), true)); // GRU layer
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
        Eigen::MatrixXf output = model.forward(inputs);

        // Compute loss
        float loss = loss_fn.forward(output, one_hot_targets);
        std::cout << "Epoch: " << epoch << " Loss: " << loss << std::endl;

        // Backward pass
        Eigen::MatrixXf grad = loss_fn.backward();  // Backprop through loss function
        model.backward(grad);                      // Backprop through layers

        // Update weights
        model.optimize();

        debugger.print_weight_change_norms();
    }
    // Validation after training
    output = model.forward(inputs);
    loss = loss_fn.forward(output, one_hot_targets);
    std::cout << "Final Loss After Training: " << loss << std::endl;

    return 0;
}

// int main() {
//     // Sample data for MLP training
//     int num_samples = 100;   // Number of samples
//     int input_size = 20;     // Input feature size
//     int output_size = 5;     // Number of classes (one-hot encoded targets)

//     // Random initialization of inputs and targets
//     Eigen::MatrixXf inputs = Eigen::MatrixXf::Random(num_samples, input_size);
//     Eigen::MatrixXf one_hot_targets(num_samples, output_size);
    
//     // Simulate one-hot encoded targets
//     for (int i = 0; i < num_samples; ++i) {
//         one_hot_targets.row(i).setZero();
//         int target_class = rand() % output_size;
//         one_hot_targets(i, target_class) = 1.0;
//     }

//     // Model setup
//     Model model;
//     model.Add(new DenseLayer(input_size, 64));
//     model.Add(new ReLU());
//     model.Add(new DenseLayer(64, 32));
//     model.Add(new ReLU());
//     model.Add(new DenseLayer(32, output_size));
//     Adam optimizer(0.001);
//     CrossEntropyLoss loss_fn;
//     model.set_optimizer(optimizer);

//     Debugger debugger;
//     debugger.track_layer(model.get_layer(2));

//     // Training loop
//     for (int epoch = 0; epoch < 10; ++epoch) {

//         debugger.save_previous_weights();

//         // Forward pass
//         Eigen::MatrixXf output = model.forward(inputs);

//         // Compute loss
//         float loss = loss_fn.forward(output, one_hot_targets);
//         std::cout << "Epoch: " << epoch << " Loss: " << loss << std::endl;

//         // Backward pass
//         Eigen::MatrixXf grad = loss_fn.backward();  // Backprop through loss function
//         model.backward(grad);                      // Backprop through layers

//         // Update weights
//         model.optimize();

//         debugger.print_weight_change_norms();
//     }

//     // model.add_callback(new EarlyStopping(5));
//     // model.Train(inputs, one_hot_targets, 50, 32, loss_fn);

//     return 0;
// }