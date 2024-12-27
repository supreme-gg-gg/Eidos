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
    auto data = NumericDataLoader("../data/mnist_mini.csv", "label", label_map).shuffle().linear_transform(1.0f / 255.0f, 0.0f).train_test_split(0.8, 32);
    // loader.print_preview(1);

    std::cout << "Data loaded from MNIST" << std::endl;

    Model model;
    model.Add(new DenseLayer(784, 128));
    model.Add(new ReLU());
    model.Add(new DenseLayer(128, 10));

    Adam optimizer(0.001);
    model.set_optimizer(optimizer);
    CrossEntropyLoss loss_fn;
    model.set_loss_function(loss_fn);
    model.add_callback(new PrintLoss(2));

    model.Train(data.training.inputs, data.training.targets, 5);
    model.Test(data.testing.inputs, data.testing.targets);
    model.Serialize("memory-test.bin");
    std::cout << "Model trained and serialized" << std::endl;
    model.Deserialize("memory-test.bin");
    std::cout << "Model deserialized" << std::endl;
    
    model.Test(data.testing.inputs, data.testing.targets);
    return 0;
}