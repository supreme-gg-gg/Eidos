#include "../include/model.h"
#include "../include/optimizer.h"
#include "../include/loss_fns.h"
#include "../include/dense_layer.h"
#include "../include/activation_fns.h"
#include <Eigen/Dense>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <filesystem>
#include <algorithm>
#include <random>

// Function to load the Iris dataset and preprocess it
void load_iris_data(const std::string& filename, Eigen::MatrixXf& targets, std::vector<std::vector<float>>& data) {
    std::ifstream file(filename);
    std::string line;

    std::getline(file, line);
    
    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string token;
        std::vector<float> row;

        // Skip the ID column and read the 4 features
        for (int i = 0; i < 4; ++i) {
            std::getline(ss, token, ',');
            row.push_back(std::stof(token));
        }

        // Read the label and convert to binary (Iris-setosa = 1, else = 0)
        std::getline(ss, token, ',');
        if (token == "Iris-setosa") {
            targets.conservativeResize(targets.rows() + 1, 1);
            targets(targets.rows() - 1, 0) = 1.0f;
        } else {
            targets.conservativeResize(targets.rows() + 1, 1);
            targets(targets.rows() - 1, 0) = 0.0f;
        }

        data.push_back(row);
    }
}

// Function to shuffle and split the data into training and testing sets
void shuffle_and_split_data(std::vector<std::vector<float>>& data, Eigen::MatrixXf& targets,
                             std::vector<std::vector<float>>& train_data, Eigen::MatrixXf& train_targets,
                             std::vector<std::vector<float>>& test_data, Eigen::MatrixXf& test_targets) {
    // Shuffle the data
    std::random_shuffle(data.begin(), data.end());

    // Split the data into training (80%) and testing (20%) sets
    int train_size = 0.8 * data.size();
    for (int i = 0; i < train_size; ++i) {
        train_data.push_back(data[i]);
        train_targets.conservativeResize(train_targets.rows() + 1, 1);
        train_targets(train_targets.rows() - 1, 0) = targets(i, 0);
    }

    for (int i = train_size; i < data.size(); ++i) {
        test_data.push_back(data[i]);
        test_targets.conservativeResize(test_targets.rows() + 1, 1);
        test_targets(test_targets.rows() - 1, 0) = targets(i, 0);
    }
}

int main() {
    // Load the Iris dataset and preprocess
    Eigen::MatrixXf targets;
    std::vector<std::vector<float>> data;
    load_iris_data("../../data/iris.csv", targets, data);

    // Shuffle and split the data into training and testing sets
    std::vector<std::vector<float>> train_data, test_data;
    Eigen::MatrixXf train_targets, test_targets;
    shuffle_and_split_data(data, targets, train_data, train_targets, test_data, test_targets);

    // Convert train_data and test_data from vector of vectors to Eigen matrices for training
    Eigen::MatrixXf train_features(train_data.size(), 4); // 4 features in Iris dataset
    for (size_t i = 0; i < train_data.size(); ++i) {
        for (size_t j = 0; j < 4; ++j) {
            train_features(i, j) = train_data[i][j];
        }
    }

    Eigen::MatrixXf test_features(test_data.size(), 4);
    for (size_t i = 0; i < test_data.size(); ++i) {
        for (size_t j = 0; j < 4; ++j) {
            test_features(i, j) = test_data[i][j];
        }
    }

    // Create and set up the model
    Model model;
    model.Add(new DenseLayer(4, 32));
    model.Add(new ReLU());
    model.Add(new DenseLayer(32, 16));
    model.Add(new ReLU());
    model.Add(new DenseLayer(16, 1));  // Output layer with 1 neuron for regression
    model.Add(new Sigmoid());  // Sigmoid activation for binary classification

    // Set optimizer and loss function
    SGD optimizer(0.01);
    BinaryCrossEntropyLoss loss_fn;
    model.set_optimizer(optimizer);

    // Training loop
    for (int epoch = 0; epoch < 100; ++epoch) {
        // Forward pass
        Eigen::MatrixXf outputs = model.forward(train_features);

        float loss = loss_fn.forward(outputs, train_targets);

        // Backward pass
        Eigen::MatrixXf grad_loss = loss_fn.backward();
        model.backward(grad_loss);

        // Optimize
        model.optimize();

        // Print loss
        std::cout << "Epoch " << epoch << " completed. Loss: " << loss << std::endl;
    }

    // Evaluate the model on the test set after training
    Eigen::MatrixXf predictions = model.forward(test_features);

    // Calculate accuracy
    int correct_predictions = 0;
    for (int i = 0; i < test_features.rows(); ++i) {
        // Convert predictions to binary (0 or 1) based on the threshold of 0.5
        if ((predictions(i, 0) > 0.5 && test_targets(i, 0) == 1.0f) || 
            (predictions(i, 0) <= 0.5 && test_targets(i, 0) == 0.0f)) {
            correct_predictions++;
        }
    }

    // Print first 10 predictions and labels
    for (int i = 0; i < std::min(100, static_cast<int>(test_features.rows())); ++i) {
        int predicted_label = predictions(i, 0) > 0.5 ? 1 : 0;
        int actual_label = test_targets(i, 0);
        std::cout << "Prediction: " << predicted_label << ", Label: " << actual_label << std::endl;
    }
    // Print test accuracy
    float accuracy = static_cast<float>(correct_predictions) / test_features.rows();
    std::cout << "Test Accuracy: " << accuracy * 100 << "%" << std::endl;

    return 0;
}