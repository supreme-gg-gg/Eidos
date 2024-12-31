#include <iostream>
#include <Eigen/Dense>
#include <Eidos/Eidos.h>

int main() {

    // Create a map to convert the labels to integers
    std::map<std::string, int> label_map;
    for (int i = 0; i < 10; ++i) {
        label_map[std::to_string(i)] = i;
    }

    // Load the MNIST dataset as images
    auto data = NumericDataLoader("../../data/mnist_mini.csv", "label", label_map).shuffle()
        .linear_transform(1.0f / 255.0f, 0.0f).train_test_split_image(28, 28, 0.8);

    // Model setup
    Model model;
    model.Add(new Conv2D(1, 32, 3, 1, 1));  // Input channels = 1, output channels = 32, kernel size = 3, stride = 1, padding = 1
    model.Add(new LeakyReLU());
    model.Add(new MaxPooling2D(2, 2));  // Kernel size = 2, stride = 2
    model.Add(new Conv2D(32, 64, 3, 1, 1));  // Input channels = 32, output channels = 64, kernel size = 3, stride = 1, padding = 1
    model.Add(new LeakyReLU());
    model.Add(new MaxPooling2D(2, 2));  // Kernel size = 2, stride = 2
    model.Add(new FlattenLayer()); // Flatten from 3D to 1D
    model.Add(new DenseLayer(3136, 128));
    model.Add(new LeakyReLU());
    model.Add(new DenseLayer(128, 10));

    Adam optimizer(0.001);
    CrossEntropyLoss loss_fn;
    PrintLoss print_loss(2);
    SaveModel save_model(model, "myModel.bin");

    model.Train(data, 20, &loss_fn, &optimizer, {&print_loss, &save_model});

    model.Serialize("myModel.bin");
    
    model.Test(data);

    return 0;
}