#include "../include/model.h"
#include "../include/optimizer.h"
#include "../include/loss_fns.h"
#include "../include/dense_layer.h"
#include "../include/activation_fns.h"
#include "../include/generic_data_loader.h"
#include <Eigen/Dense>
#include <iostream>

int main() {

    GenericDataLoader loader;
    std::vector<Eigen::MatrixXf> features;
    std::vector<std::string> labels;
    loader.load_data("../../data/iris.csv", features, labels);

    std::map<std::string, int> label_to_index = {{"Iris-setosa", 0}, {"Iris-versicolor", 1}, {"Iris-virginica", 2}};
    std::vector<Eigen::MatrixXf> one_hot_labels;

    for (const auto& label : labels) {
        Eigen::MatrixXf one_hot = Eigen::MatrixXf::Zero(1, 3);
        one_hot(0, label_to_index[label]) = 1.0;
        one_hot_labels.push_back(one_hot);
    }

    // Split the data into training and testing sets
    std::vector<Eigen::MatrixXf> train_features, test_features;
    std::vector<Eigen::MatrixXf> train_labels, test_labels;

    float train_ratio = 0.8;
    int train_size = static_cast<int>(features.size() * train_ratio);

    for (int i = 0; i < features.size(); ++i) {
        if (i < train_size) {
            train_features.push_back(features[i]);
            train_labels.push_back(one_hot_labels[i]);
        } else {
            test_features.push_back(features[i]);
            test_labels.push_back(one_hot_labels[i]);
        }
    }

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
    model.Add(new DenseLayer(4, 16));
    model.Add(new Tanh());
    model.Add(new DenseLayer(16, 32));
    model.Add(new Tanh());
    model.Add(new DenseLayer(32, 16));
    model.Add(new Tanh());
    model.Add(new DenseLayer(16, 8));
    model.Add(new Tanh());
    model.Add(new DenseLayer(8, 3));  // Output layer with 1 neuron for regression

    // Set optimizer and loss function
    Adam optimizer(0.001);
    CrossEntropyLoss loss_fn;
    
    // Training
    model.Train(inputs, targets, 300, 32, loss_fn, &optimizer);

    // Testing
    model.Test(test_inputs, test_targets, loss_fn);

    return 0;
}