# Additional Layer Types

This tutorial goes over regularisation layers, CNN layers, and RNN layers. It assumes you have read the basic tutorial and are familiar with the basic concepts of these layers.

## Convolutional Neural Networks

Convolutional Neural Networks (CNNs) are a class of deep neural networks that are primarily used for image classification and object detection tasks. CNNs are designed to automatically and adaptively learn spatial hierarchies of features from image data.

A simple implementation follows:

```cpp
Conv2D conv1(1, 32, 3, 1, 1); // 1 input channel, 32 output channels, 3x3 kernel, stride 1, padding 1
ReLU relu1;
MaxPooling2D pool1(2, 2); // 2x2 kernel, stride 2

Conv2D conv2(32, 64, 3, 1, 1); // 32 input channels, 64 output channels, 3x3 kernel, stride 1, padding 1
ReLU relu2;
MaxPooling2D pool2(2, 2); // 2x2 kernel, stride 2

FlattenLayer flatten; // Flatten from 3D to 1D

DenseLayer fc(7 * 7 * 64, 10); // 7x7x64 input size, 10 output size
```

You do not necessarily need to use the `FlattenLayer` if you are using a custom training loop. The Tensor method `flatten()` can be used instead if you wish to reshape to other dimensions.

## Recurrent Neural Networks

Recurrent Neural Networks (RNNs) are a class of neural networks that are designed to handle sequential data. RNNs are capable of capturing temporal dependencies in data and are commonly used for tasks such as time series prediction, speech recognition, and natural language processing.

RNN layer will dynamically unroll the network for the length of the input sequence. However, the input and output sizes are provided. Additionally, you can specify if the model should output at each time step or only at the end.

> Output layer is assumed to be a dense layer, but this can easily be customized.

```cpp
// 10 input size, 20 hidden size, 5 output size, Tanh activation, output at each time step
RNNLayer rnn(10, 20, 5, new Tanh(), true);

Tensor input({1, 10, 10}); // 10 time steps, 10 input size, no batching

auto output = rnn.forward(input); // Output is of size {1, 5, 10}
```

A GRU (Gated Recurrent Unit) layer uses a more advanced architecture that leverages gating mechanisms to better capture long-term dependencies in data and avoid the vanishing gradient problem. **You are required to provide activation functions for the update and reset gates (usually `Sigmoid()`), as well as the candidate hidden state (usually `Tanh()`).**

```cpp
// 10 input size, 20 hidden size, 5 output size, output at each time step
GRULayer gru(10, 20, 5, new Sigmoid(), new Tanh(), true);
```

## Regularisation Layers

Regularisation is a technique used to prevent overfitting in machine learning models. Regularisation layers can be added to a model to apply techniques such as dropout and batch normalisation.

> During training and testing, the model will automatically switch between training and evaluation modes to apply the regularisation techniques correctly **(except for custom training loops)**.

You must toggle the model between training and evaluation modes if you are using a custom training loop by calling `model.set_train()` and `model.set_inference()`.

### Dropout

Dropout is a technique used to prevent overfitting in neural networks. It works by randomly setting a fraction of the input units to zero during training, which helps to prevent the network from relying too heavily on any one input feature.

```cpp
Dropout dropout(0.5); // Dropout rate of 0.5
```

### Batch Normalisation

Batch Normalisation is a technique used to improve the training of deep neural networks. It works by normalising the input to each layer of the network to have zero mean and unit variance. This helps to stabilise the training process and can lead to faster convergence and better generalisation.

```cpp
BatchNorm batch_norm(128); // Input size of 128
```

These layers can be used in combination with other layers to build more complex models. For example, you can add dropout layers after each dense layer in a model to prevent overfitting, or add batch normalisation layers to improve the training process.

```cpp
Model model;
model.Add(new DenseLayer(4, 32));
model.Add(new BatchNorm(32));
model.Add(new LeakyReLU());
model.Add(new Dropout(0.2));
model.Add(new DenseLayer(32, 16));
model.Add(new BatchNorm(16));
model.Add(new LeakyReLU());
model.Add(new DenseLayer(16, 3));
```

## Conclusion

In this tutorial, we covered additional layer types such as CNNs, RNNs, and regularisation layers. These layers can be used to build more complex models and improve the performance of your machine learning algorithms. You should now move on to the final [tutorial](./utilities.md) where we cover more advanced utilities.
