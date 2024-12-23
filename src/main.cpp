#include "../include/console.hpp"
#include "../include/model.h"
#include "../include/optimizer.h"
#include "../include/loss_fns.h"
#include "../include/layers.h"
#include "../include/activation_fns.h"
#include "../include/preprocessors.h"
#include "../include/callback.h"
#include "../include/debugger.hpp"
#include <Eigen/Dense>
#include <iostream>

int main() {
    Console::config(false);
    Timer timer;

    NumericDataLoader loader("../../data/mnist_mini.csv", "label");
    loader.shuffle();
    auto data = loader.train_test_split_image(28, 28, 0.8);
    // loader.print_preview(1);

    std::cout << "Data loaded from MNIST" << std::endl;
    
    // Create and set up the model
    Model model;
    model.Add(new Conv2D(1, 3, 3, 1, 1));
    model.Add(new Tanh());
    model.Add(new MaxPooling2D(2, 2));
    model.Add(new Conv2D(3, 3, 3, 1, 1));
    model.Add(new Tanh());
    model.Add(new MaxPooling2D(2, 2));
    model.Add(new FlattenLayer());
    model.Add(new DenseLayer(147, 32));
    model.Add(new Tanh());
    model.Add(new Dropout(0.3));
    model.Add(new DenseLayer(32, data.num_classes()));

    // // Set optimizer and loss function
    Adam optimizer(0.001);
    CrossEntropyLoss loss_fn;
    // PrintLoss print_loss(5);
    model.set_optimizer(optimizer);
    model.set_loss_function(loss_fn);

    model.set_train();
    for (int epoch = 0; epoch < 20; ++epoch) {
        float total_loss = 0.0;
        for (size_t i = 0; i < data.training.inputs.size(); ++i) {
            Tensor output = model.forward(data.training.inputs[i]);
            float loss = loss_fn.forward(output, data.training.targets[i]);
            total_loss += loss;
            model.backward();
            model.optimize();
        }
        std::cout << "Epoch " << epoch << " | Average Loss: " << total_loss / data.training.inputs.size() << std::endl;
    }

    // Training
    // model.Train(data, 5, loss_fn, &optimizer);

    // Testing
    model.Test(data, loss_fn);

    return 0;
}