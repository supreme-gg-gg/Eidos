#include "../include/model.h"
#include "../include/optimizer.h"
#include "../include/loss_fns.h"
#include "../include/dense_layer.h"
#include "../include/activation_fns.h"
#include "../include/generic_data_loader.h"
#include <Eigen/Dense>
#include <iostream>

int main() {
    std::string csvPath;
    std::cout << "Enter the path to the CSV file: ";
    std::getline(std::cin, csvPath);
    GenericDataLoader loader;
    std::vector<Eigen::MatrixXf> features;
    std::vector<std::string> labels;
    loader.load_data(csvPath, features, labels);
    
    std::vector<Eigen::MatrixXf> train_features, test_features;
    std::vector<std::string> train_string_labels, test_string_labels;
    std::vector<Eigen::MatrixXf> train_labels, test_labels;

    loader.split_data(features, labels, train_features, train_string_labels, test_features, test_string_labels, 0.8);
    
    //std::map<std::string, int> label_to_index = {{"Iris-setosa", 0}, {"Iris-versicolor", 1}, {"Iris-virginica", 2}};
    std::map<std::string, int> label_to_index = {{"0", 0}, {"1", 1}, {"2", 2}, {"3", 3}, {"4", 4}, {"5", 5}, {"6", 6}, {"7", 7}, {"8", 8}, {"9", 9}};
    loader.convert_to_one_hot(train_string_labels, train_labels, label_to_index);
    loader.convert_to_one_hot(test_string_labels, test_labels, label_to_index);
    
    // Convert vectors to Eigen matrices for easier manipulation
    Eigen::MatrixXf inputs(train_features.size(), train_features[0].size());
    Eigen::MatrixXf targets(train_labels.size(), train_labels[0].size());

    for (int i = 0; i < train_features.size(); ++i) {
        inputs.row(i) = train_features[i];
        targets.row(i) = train_labels[i];
    }

    // batch size = sample size
    // SET batch size = 32

    // Multiple batches 32*4 
    // 4 batches -> 32*4 samples in total -> 4 iterations of the training loop

    Eigen::MatrixXf test_inputs(test_features.size(), test_features[0].size());
    Eigen::MatrixXf test_targets(test_labels.size(), test_labels[0].size());

    for (int i = 0; i < test_features.size(); ++i) {
        // Also normalize the input and target values
        test_inputs.row(i) = test_features[i];
        test_targets.row(i) = test_labels[i];
    }

    // Print the first 5 rows of the input and target
    std::cout << "First 5 rows of inputs:" << std::endl;
    for (int i = 0; i < 5 && i < inputs.rows(); ++i) {
        std::cout << inputs.row(i) << std::endl;
    }

    std::cout << "First 5 rows of targets:" << std::endl;
    for (int i = 0; i < 5 && i < targets.rows(); ++i) {
        std::cout << targets.row(i) << std::endl;
    }

    // Create and set up the model
    Model model;
    model.Add(new DenseLayer(784, 16));
    model.Add(new Tanh());
    model.Add(new DenseLayer(16, 16));
    model.Add(new Tanh());
    model.Add(new DenseLayer(16, 16));
    model.Add(new Tanh());
    model.Add(new DenseLayer(16, 10));   // Output layer with 1 neuron for regression

    // Set optimizer and loss function
    Adam optimizer(0.0005);
    CrossEntropyLoss loss_fn;
    
    // Training
    model.Train(inputs, targets, 200, 32, loss_fn, &optimizer);

    // Testing
    model.Test(test_inputs, test_targets, loss_fn);

    return 0;
}