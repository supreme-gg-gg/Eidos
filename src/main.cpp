#include "../include/console.hpp"
#include "../include/model.h"
#include "../include/optimizer.h"
#include "../include/loss_fns.h"
#include "../include/layers.h"
#include "../include/activation_fns.h"
#include "../include/preprocessors/numeric_data_loader.h"
#include "../include/callback.h"
#include "../include/debugger.hpp"
#include <Eigen/Dense>
#include <iostream>

int main() {
    Console::config(true);
    Timer timer;

    std::string csvPath;
    std::cout << "Enter the path to the CSV file: ";
    std::getline(std::cin, csvPath);
    NumericDataLoader loader(csvPath, "label");
    loader.shuffle();
    auto data = loader.train_test_split_image(28, 28, 0.8f);
    // Create and set up the model
    Model model;
    model.Add(new DenseLayer(data.num_features(), 32));
    model.Add(new BatchNorm(32));
    model.Add(new LeakyReLU());
    model.Add(new Dropout(0.2));
    model.Add(new DenseLayer(32, 16));
    model.Add(new BatchNorm(16));
    model.Add(new LeakyReLU());
    model.Add(new DenseLayer(16, 8));
    model.Add(new BatchNorm(8));
    model.Add(new LeakyReLU());
    model.Add(new DenseLayer(8, data.num_classes()));

    // // Set optimizer and loss function
    Adam optimizer(0.01);
    CrossEntropyLoss loss_fn;

    // // Training
    model.Train(data.training.inputs, data.training.targets, 20, loss_fn, &optimizer);

    // // Testing
    model.Test(data.testing.inputs, data.testing.targets, loss_fn);

    return 0;
}