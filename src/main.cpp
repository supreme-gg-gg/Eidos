#include "../include/model.h"
#include "../include/optimizer.h"
#include "../include/loss_fns.h"
#include "../include/dense_layer.h"
#include "../include/activation_fns.h"
#include "../include/generic_data_loader.h"
#include "../include/regularization.h"
#include <Eigen/Dense>
#include <iostream>
#include <chrono>

class Timer {
public:
    Timer() : start_time_point(std::chrono::high_resolution_clock::now()) {}

    ~Timer() {
        auto end_time_point = std::chrono::high_resolution_clock::now();
        auto start = std::chrono::time_point_cast<std::chrono::milliseconds>(start_time_point).time_since_epoch().count();
        auto end = std::chrono::time_point_cast<std::chrono::milliseconds>(end_time_point).time_since_epoch().count();
        auto duration = end - start;
        std::cout << "Duration: " << duration << " ms" << std::endl;
    }

private:
    std::chrono::time_point<std::chrono::high_resolution_clock> start_time_point;
};

int main() {

    Timer timer;

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

    // std::map<std::string, int> label_to_index = {{"Iris-setosa", 0}, {"Iris-versicolor", 1}, {"Iris-virginica", 2}};
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

    Eigen::MatrixXf test_inputs(test_features.size(), test_features[0].size());
    Eigen::MatrixXf test_targets(test_labels.size(), test_labels[0].size());

    for (int i = 0; i < test_features.size(); ++i) {
        // Also normalize the input and target values
        test_inputs.row(i) = test_features[i];
        test_targets.row(i) = test_labels[i];
    }

    std::cout << "First 5 rows of targets:" << std::endl;
    for (int i = 0; i < 5 && i < targets.rows(); ++i) {
        std::cout << targets.row(i) << std::endl;
    }

    // Create and set up the model
    Model model;
    model.Add(new DenseLayer(784, 128));
    model.Add(new BatchNorm(128));
    model.Add(new ReLU());
    model.Add(new Dropout(0.2));
    model.Add(new DenseLayer(128, 64));
    model.Add(new BatchNorm(64));
    model.Add(new ReLU());
    model.Add(new Dropout(0.2));
    model.Add(new DenseLayer(64, 10));

    // Set optimizer and loss function
    Adam optimizer(0.01);
    CrossEntropyLoss loss_fn;

    // Training
    model.Train(inputs, targets, 20, 256, loss_fn, &optimizer);

    // model.set_optimizer(optimizer);
    // for (int epoch = 0; epoch < 100; ++epoch) {
    //     Eigen::MatrixXf outputs = model.forward(inputs);
    //     float loss = loss_fn.forward(outputs, targets);
    //     Eigen::MatrixXf grad_loss = loss_fn.backward();
    //     model.backward(grad_loss);
    //     model.optimize();

    //     std::cout << "Epoch " << epoch << " completed. Loss: " << loss << std::endl;
    // }

    // Testing
    model.Test(test_inputs, test_targets, loss_fn);

    return 0;
}

// int main() {
//     // Sample data
//     Eigen::MatrixXf inputs = Eigen::MatrixXf::Random(100, 20);
//     Eigen::MatrixXf targets = Eigen::MatrixXf::Random(100, 5);
//     Eigen::MatrixXf one_hot_targets = Eigen::MatrixXf::Zero(targets.rows(), 5);
//     for (int i = 0; i < targets.rows(); ++i) {
//         int maxIndex;
//         targets.row(i).maxCoeff(&maxIndex);
//         one_hot_targets(i, maxIndex) = 1.0;
//     }
//     targets = one_hot_targets;

//     Model model;
//     model.Add(new DenseLayer(20, 32));
//     model.Add(new BatchNorm(32));
//     model.Add(new LeakyReLU());
//     model.Add(new Dropout(0.2));
//     model.Add(new DenseLayer(32, 64));
//     model.Add(new BatchNorm(64));
//     model.Add(new LeakyReLU());
//     model.Add(new DenseLayer(64, 5));

//     Adam optimizer(0.01);
//     CrossEntropyLoss loss_fn;

//     model.set_optimizer(optimizer);

//     for (int epoch = 0; epoch < 20; ++epoch) {
//         Eigen::MatrixXf output = model.forward(inputs);
//         float loss = loss_fn.forward(output, targets);
//         std::cout << "Epoch: " << epoch << " Loss: " << loss << std::endl;
//         Eigen::MatrixXf grad = loss_fn.backward();
//         model.backward(grad);
//         model.optimize();
//     }
// }