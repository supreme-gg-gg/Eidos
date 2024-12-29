# Advanced Utilities

This is the last section that will cover advanced utilities provided by Eidos. These utilities are designed to help you perform advanced tasks including debugging custom setup and implementing custom classes.

## Callbacks

Callbacks make use of training and validation loops to perform custom tasks. This is useful for logging, saving models, and early stopping. The following is an example of a callback that logs the loss and accuracy of the model:

We support the following callbacks:

- `EarlyStopping`: Stops training when loss does not improve for a certain number of epochs.
- `PrintLoss`: Prints the loss at specified intervals.
- `SaveModel`: Saves the model at specified intervals as training checkpoints using `Model::Serialize()`.

To use each callback, simply add it to the `Model` object or pass it as a parameter to the `Train` function which invokes and manages all callbacks automatically. It is not recommended to use custom training loops with built-in callbacks.

```cpp
EarlyStopping early_stopping(5);
PrintLoss print_loss(1);

model.Train(data.training.inputs, data.training.targets, 5, &optimizer, &loss_fn, {&early_stopping, &print_loss});

// alternatively
model.add_callback(&early_stopping);
model.add_callback(&print_loss);
model.Train(data.training.inputs, data.training.targets, 5, &optimizer, &loss_fn);
```

Creating custom callbacks is easy by inheriting from the `Callback` class and implementing the `on_epoch_end` function.

## Debugger

The debugger utility in `debugger.hpp` provides a simple toolkit for you to debug your model and custom layers. For example, you can track layers to log weight change norms and gradient norms to ensure that your model is learning correctly.

A built in `Timer` class is also provided to measure the time taken for the training. You can use it by simply creating a timer object which logs the time in its destructor.

```cpp
// Create a debugger object
Debugger debugger;
debugger.track_layer(model.get_layer(0)); // Track the first layer

Timer timer;

for (int i = 0; i < 5; ++i) {
    auto output = model.Forward(data.training.inputs);
    auto loss = loss_fn.Forward(output, data.training.targets);
    debugger.print_gradient_norms();
    model.backward()
    model.optimize();
    debugger.print_weight_change_norms();
}
```

## Console

The `Console` namespace provides a configurable logging tool with multiple log levels. Use it to print messages to the console with specific behaviors and filters. The following log levels are supported:

| **Flag**  | **Description**                           |
| --------- | ----------------------------------------- |
| `INFO`    | General information messages.             |
| `WARNING` | Warnings about potential issues.          |
| `ERROR`   | Critical errors.                          |
| `DEBUG`   | Debugging messages (requires debug mode). |
| `WORSHIP` | Custom special messages.                  |

To use the console, simply include the header and call the `Console::log` function with the desired log level and message.

```cpp
Console::log("Initialization complete.", Console::INFO);
Console::log("Potential issue detected.", Console::WARNING);
Console::log("Critical failure occurred.", Console::ERROR);
```

To view the system debug messages:

```cpp
Console::config(true); // turn on debug messages
```

## Conclusion

Congratulations! You have now learned about all the functionalities provided by Eidos! This is a customizable library, meaning you can extend it to suit your needs. We hope you enjoy using Eidos and find it helpful in your machine learning projects. If you have any questions or feedback, please feel free to reach out to us.
