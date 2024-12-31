#include <Eigen/Dense>
#include <Eidos/Eidos.h>
#include <iostream>

/*
This program demonstrates how to train a MLP model on the MNIST dataset 
using the Eidos library. The MNIST dataset is a simple dataset that contains
handwritten digits from 0 to 9. The goal is to classify the digits based on the
pixel values. This model architecture achieved 97% accuracy on the test set.

The model also demonstrates how to integrate regularization techniques like
Batch Normalization, Dropout, and LeakyReLU activation function.

The model is trained for 20 epochs and serialized to a file. The model is then
deserialized and tested on the test set to verify the accuracy.

The saved model can be used for inference on new data or further training by 
creating a new model and loading the weights then calling the forward method.

Released under the MIT License https://opensource.org/licenses/MIT
*/

int main() {
    Console::config(true);

    std::map<std::string, int> label_map;
    for (int i = 0; i < 10; ++i) {
        label_map[std::to_string(i)] = i;
    }

    auto data = NumericDataLoader("../../data/mnist_train.csv", "label", label_map).shuffle()
        .linear_transform(1.0f / 255.0f, 0.0f).train_test_split(0.8, 32);

    Model model;
    model.Add(new DenseLayer(784, 256));
    model.Add(new BatchNorm(256));
    model.Add(new LeakyReLU());
    model.Add(new Dropout(0.2));
    model.Add(new DenseLayer(256, 64));
    model.Add(new BatchNorm(64));
    model.Add(new LeakyReLU());
    model.Add(new DenseLayer(64, 32));
    model.Add(new BatchNorm(32));
    model.Add(new LeakyReLU());
    model.Add(new DenseLayer(32, 10));

    Adam optimizer(0.001);
    model.set_optimizer(optimizer);

    CrossEntropyLoss loss_fn;
    model.set_loss_function(loss_fn);

    model.add_callback(new PrintLoss(2));

    model.Train(data.training.inputs, data.training.targets, 20);
    model.Test(data.testing.inputs, data.testing.targets);

    model.Serialize("myModel.bin");
    std::cout << "Model trained and serialized" << std::endl;

    model.Deserialize("myModel.bin");
    std::cout << "Model deserialized" << std::endl;
    
    model.Test(data.testing.inputs, data.testing.targets);

    return 0;
}