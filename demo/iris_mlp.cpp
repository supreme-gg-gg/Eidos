#include <iostream>
#include <Eigen/Dense>
#include <Eidos/Eidos.h>

/*
This example demonstrates how to train a simple 
Multi-Layer Perceptron (MLP) model using the Eidos library.

We will use the Iris dataset, which is a simple dataset
that contains 4 features and 3 classes. The goal is to
classify the species of Iris flowers based on the features.

We will use a custom training loop to show how you can train
a model without using the built-in training methods. This allow
you to have more control over the training process for complicated usage.
*/

int main() {
    
    std::map<std::string, int> label_to_index = {{"Iris-setosa", 0}, 
        {"Iris-versicolor", 1}, {"Iris-virginica", 2}};

    NumericDataLoader loader = NumericDataLoader("iris.csv", "Species", label_to_index);
    loader.print_preview(5);

    auto data = loader.shuffle().train_test_split(0.8, 4);

    // Model setup
    Model model;
    model.Add(new DenseLayer(4, 32));
    model.Add(new BatchNorm(32));
    model.Add(new LeakyReLU());
    model.Add(new Dropout(0.2));
    model.Add(new DenseLayer(32, 16));
    model.Add(new BatchNorm(16));
    model.Add(new LeakyReLU());
    model.Add(new DenseLayer(16, 3)); 

    Adam optimizer(0.001);
    CrossEntropyLoss loss_fn;

    model.set_optimizer(optimizer);
    model.set_loss_function(loss_fn);

    model.set_train();
    for (int epoch = 0; epoch < 20; ++epoch) {
        for (int i = 0; i < data.training.inputs.rows(); ++i) {
            Tensor output = model.forward(data.training.inputs[i]);
            float loss = loss_fn.forward(output, data.training.targets[i]);
            std::cout << "Epoch: " << epoch << " Loss: " << loss << std::endl;
            model.backward();
            model.optimize();
        }
    }

    model.Test(data.testing.inputs, data.testing.targets, loss_fn);

    model.Serialize("iris_model.bin", true, true, false);

    return 0;
}