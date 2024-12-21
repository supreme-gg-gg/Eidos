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
    NumericDataLoader loader(csvPath, "Species");
    loader.shuffle().z_score_normalize();
    loader.print_preview();
    auto data = loader.train_test_split(0.8f, 1);
    // Create and set up the model
    /*
    Model model;
    model.Add(new DenseLayer(data.num_features(), 256));
    model.Add(new BatchNorm(256));
    model.Add(new LeakyReLU());
    model.Add(new Dropout(0.2));
    model.Add(new DenseLayer(256, 64));
    model.Add(new BatchNorm(64));
    model.Add(new LeakyReLU());
    model.Add(new DenseLayer(64, 32));
    model.Add(new BatchNorm(32));
    model.Add(new LeakyReLU());
    model.Add(new DenseLayer(32, data.num_classes())); 
    */
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

    // Set optimizer and loss function
    Adam optimizer(0.01);
    CrossEntropyLoss loss_fn;

    // Training
    model.Train(data.training.inputs, data.training.targets, 20, loss_fn, &optimizer);

    // Testing
    model.Test(data.testing.inputs, data.testing.targets, loss_fn);

    return 0;
}

// int main() {

//     Tensor input_tensor = Tensor(1, 10, 10); // Example dimensions for RNN input
//     Tensor target_tensor = Tensor(1, 10, 3); // Example dimensions for RNN target

//     Model model;
//     model.Add(new RNNLayer(10, 25, 3, new Sigmoid())); // Input size 8, hidden size 16, sequence length 5

//     Adam optimizer(0.001);
//     CrossEntropyLoss loss_fn;

//     model.set_optimizer(optimizer);

//     for (int epoch = 0; epoch < 2; ++epoch) {
//         Tensor output = model.forward(input_tensor);
//         float loss = loss_fn.forward(output, target_tensor);
//         std::cout << "Epoch: " << epoch << " Loss: " << loss << std::endl;

//         Tensor grad = loss_fn.backward();
//         model.backward(grad);
//         model.optimize();
//     }

//     return 0;
// }