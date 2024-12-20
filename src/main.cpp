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

    Tensor input_tensor = Tensor(1, 10, 10); // Example dimensions for RNN input
    Tensor target_tensor = Tensor(1, 10, 3); // Example dimensions for RNN target

    Model model;
    model.Add(new RNNLayer(10, 25, 3, new Sigmoid())); // Input size 8, hidden size 16, sequence length 5

    Adam optimizer(0.001);
    CrossEntropyLoss loss_fn;

    model.set_optimizer(optimizer);

    for (int epoch = 0; epoch < 2; ++epoch) {
        Tensor output = model.forward(input_tensor);
        float loss = loss_fn.forward(output, target_tensor);
        std::cout << "Epoch: " << epoch << " Loss: " << loss << std::endl;

        Tensor grad = loss_fn.backward();
        model.backward(grad);
        model.optimize();
    }

    return 0;
}