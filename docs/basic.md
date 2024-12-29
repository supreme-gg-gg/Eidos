# Basic Usage of Eidos

Welcome to the first tutorial of Eidos! This, as well as the following tutorials, will guide you through the basic usage of Eidos. The purpose of these tutorials is to expose the user to the basic concepts of Eidos and how to use them. For experienced users, please refer to `demo/` for direct application examples.

This section will guide you through the basic usage of Eidos. We will show you how to create a simple MLP model and train it on the MNIST dataset.

## Model

A model consists of layers. You can add layers to the model using the `Add` method. This allow you to use forward and backward methods on the model to compute the loss and gradients automatically.

```cpp
Model model;
model.Add(new DenseLayer(784, 128)); // Input size is 784 and output size is 128
model.Add(new ReLU()); // Activation function
model.Add(new DenseLayer(128, 10)); // Input size is 128 and output size is 10
```

A model has members optimizer and loss function and callbacks (refer to seprate doc). You can set them by:

```cpp
model.set_optimizer(new Adam(0.001)); // Learning rate is 0.001
model.set_loss_fn(new CrossEntropyLoss());
```

The model has a `forward` method that takes in the input and returns the output. It also has a `backward` method that computes the gradients of the loss with respect to the model parameters.

```cpp
Tensor output = model.forward(input);
model.backward();
```

> Note that while the entire library is built on top of custom Tensors, usually you will not need to interact with them directly. The preprocessing functionalities provide powerful abstractions to wrap your data in a format that can be directly fed into the model.

## Layers and Activations

A layer is a basic building block of a model. Both activation functions and all model layers are derived from the `Layer` class. We support the following layers:

- Dense Layer: `DenseLayer`
- Convolutional Layer: `Conv2D`
- Max Pooling Layer: `MaxPooling2D`
- Average Pooling Layer: `AveragePooling2D`
- Flatten Layer: `FlattenLayer`
- Dropout Layer: `Dropout`
- Batch Normalization Layer: `BatchNorm`
- RNN Layer: `RNNLayer`
- GRU Layer: `GRULayer`

Convolutional network and recurrent networks will be discussed in later tutorials.

We support the following activation functions:

- ReLU: `ReLU`
- Leaky ReLU: `LeakyReLU`
- Sigmoid: `Sigmoid`
- Tanh: `Tanh`
- Softmax: `Softmax`

### Dense Layer

A dense layer is a fully connected layer where each neuron is connected to every neuron in the previous layer. You can create a dense layer by passing the input size and output size. You can also add an activation function to the layer to introduce non-linearity.

```cpp
DenseLayer layer(784, 128); // Input size is 784 and output size is 128
LeakyReLU activation(0.01); // Leaky ReLU with negative slope of 0.01
```

You can implement your own layer by inheriting from the `Layer` class and implementing the `forward` and `backward` methods.

## Optimizer

An optimizer is used to update the parameters of the model. The library provides the basic SGD and Adam optimizers. You can create an optimizer by passing the learning rate.

```cpp
Adam optimizer(0.001); // default beta1 = 0.9, beta2 = 0.999, epsilon = 1e-8
SGD optimizer(0.001);
```

Each optimizer has a `optimize` method that takes in the layer and update its parameters by retrieving the gradients stored in the layer. This is done on all layers when you call the `optimize` method on the model.

```cpp
optimizer.optimize(layer); // invokes layer.get_weights() and updates layer.weights
```

You can implement your own optimizer by inheriting from the `Optimizer` class and implementing the `optimize` method.

## Loss Function

A loss function is used to compute the loss given the output of the model and the target. The library provides the basic loss functions:

- Mean Squared Error: `MSELoss`
- Cross Entropy Loss
  - With logits: `CrossEntropyLoss`
  - With probabilities: `CategoricalCrossEntropyLoss`
- Binary Cross Entropy Loss: `BinaryCrossEntropyLoss`

Loss function contain a `forward` method that takes in the output of the model and the target and returns the loss. It also contains a `backward` method that computes the gradients of the loss with respect to the output.

```cpp
CrossEntropyLoss loss_fn;
auto loss = loss_fn.forward(output, target);
loss_fn.backward(); // computes the gradients
```

You can implement your own loss function by inheriting from the `Loss` class and implementing the `forward` and `backward` methods.

## Data Loader

A data loader is used to load the data and split it into training and testing sets. The library provides the `NumericDataLoader` class to load numeric data. Other data functions are explored more extensively in subsequent tutorials. You can create a data loader by passing the file name, label column, and label map.

```cpp
std::map<std::string, int> label_map; // then create your own label mapping...
NumericDataLoader data_loader("mnist_train.csv", "label", label_map);
auto data = data_loader.train_test_split(0.8, 32); // 80% training and 32 batch size
```

## Convenience API

The library provides a convenience API to train and test the model with more abstraction. You can train the model by passing the inputs, targets, number of epochs, optimizer, and loss function. The training data passed in must be either a `Tensor` or `ImageInputData` object. You do not need to pass in references to the optimizer and loss function if you have already set them in the model.

```cpp
model.Train(data.training.inputs, data.training.targets, 5, &optimizer, &loss_fn);
model.Train(data, 5, &optimizer, &loss_fn); // it also supports ImageInputData class
```

This is similar with testing:

```cpp
model.Test(data.testing.inputs, data.testing.targets, &loss_fn); // outputs accuracy and test loss
```

## Serialization

We save model as binary file using `Serialize` method. You can load the model using `Deserialize` method. By default, only weights are saved (i.e. no optimizer, loss function, etc.). However, you can set `weights_only` to false to save the entire configuration. By default, the model architecture is saved as a text file with the same name as the binary file.

```cpp
model.Serialize("myModel.bin"); // generates myModel.bin and myModel.txt
model.Deserialize("myModel.bin"); // loads the model in place on the model object
```

## Custom Training Loop

The forward pass computes the output of the model given the input. The backward pass computes the gradients of the loss with respect to the model parameters. The gradients are stored in the layers and can be used to update the parameters.

We can use this to construct a sample custom training loop:

```cpp
// Iterate over the epochs
for (int epoch = 0; epoch < 5; ++epoch) {

    // Iterate over the training batches
    for (int i = 0; i < data.training.inputs.size(); ++i) {

        // Forward pass
        auto output = model.forward(data.training.inputs[i]);

        // Compute loss
        auto loss = loss_fn.forward(output, data.training.targets[i]);

        // Backward pass (invokes loss_fn.backward and model.backward)
        model.backward();

        // Update parameters
        model.optimize();
    }
}
```

## Conclusion

This is how you can create a simple MLP model and train it on the MNIST dataset after putting it all together:

### Training

```cpp
#include <Eidos/Eidos.h>

int main() {

    // Create a label map for the data loader
    std::map<std::string, int> label_map;
    for (int i = 0; i < 10; ++i) {
        label_map[std::to_string(i)] = i;
    }

    // Load data and split into training and testing
    auto data = NumericDataLoader("mnist_train.csv", "label", label_map).train_test_split(0.8, 32);

    // Create model and add fully connected layers
    Model model;
    model.Add(new DenseLayer(784, 128));
    model.Add(new ReLU());
    model.Add(new DenseLayer(128, 10));

    // Create optimizer and loss function
    Adam optimizer(0.001);
    CrossEntropyLoss loss_fn;

    // Train and test the model using convenience API
    model.Train(data.training.inputs, data.training.targets, 5, &optimizer, &loss_fn);
    model.Test(data.testing.inputs, data.testing.targets, &loss_fn);

    // Save the model
    model.Serialize("myModel.bin");
}
```

### Inferencing

```cpp
#include <Eidos/Eidos.h>

int main() {

    // Load data and split into training and testing
    auto data = NumericDataLoader("mnist_test.csv", "label", label_map).train_test_split(0.8, 32);

    // Load the model
    Model model;
    model.Deserialize("myModel.bin");

    // Test the model
    model.Test(data.testing.inputs, data.testing.targets, &loss_fn);
}
```

For next steps, move on to learning more about data preprocessing at [Data Preprocessing](./data.md).
