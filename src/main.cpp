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
    Console::config(true);
    Timer timer;
    std::map<std::string, int> label_map;
    for (int i = 0; i < 10; ++i) {
        label_map[std::to_string(i)] = i;
    }
    auto data = NumericDataLoader("../data/mnist_small.csv", "label", label_map).shuffle().linear_transform(1.0f / 255.0f, 0.0f).train_test_split_image(28, 28, 0.8);
    // loader.print_preview(1);

    std::cout << "Data loaded from MNIST" << std::endl;
    
    // Create and set up the model
    Model model;
    model.Add(new Conv2D(1, 16, 3, 1, 1));
    model.Add(new Tanh());
    model.Add(new MaxPooling2D(2, 2));
    model.Add(new Conv2D(16, 16, 3, 1, 1));
    model.Add(new Tanh());
    model.Add(new MaxPooling2D(2, 2));
    model.Add(new FlattenLayer());
    model.Add(new DenseLayer(784, 128));
    model.Add(new Tanh());
    model.Add(new Dropout(0.3));
    model.Add(new DenseLayer(128, 32));
    model.Add(new Tanh());
    model.Add(new DenseLayer(32, 10));
    
    Adam optimizer(0.00001);
    model.set_optimizer(optimizer);
    CrossEntropyLoss loss_fn;
    model.set_loss_function(loss_fn);
    std::vector<Callback*> callbacks = {new PrintLoss(1), new SaveModel(model, "mnist_digits_small-cnn_mini.bin", 1), new EarlyStopping(10.0f)};
    
    // Testing (to check if the model is working)
    model.Test(data);
    
    // Training the model
    model.Train(data, 50, &loss_fn, &optimizer, callbacks);
    
    model.Test(data);
    
    return 0;
}