#include "../include/model.h"
#include "../include/console.hpp"
#include "../include/csvparser.h"
#include "../include/layer.h"
#include "../include/activation_fns.h"
#include "../include/optimizer.h"
#include "../include/layers/flatten_layer.h"
#include "../include/tensor.hpp"
#include <fstream>
#include <filesystem>
#include <string>
#include <vector>
#include <algorithm>
#include <random>
#include <Eigen/Dense>

void Model::Add(Layer* layer) {
    this->layers.emplace_back(layer); // Wraps raw pointer in a unique_ptr
}

void Model::add_callback(Callback* callback) {
    callbacks.push_back(callback);
}

Layer* Model::get_layer(size_t index) const {
    if (index >= this->layers.size()) {
        throw std::out_of_range("Layer index out of range");
    }
    return layers[index].get();  // Access raw pointer from unique_ptr
}

size_t Model::num_layers() const {
    return this->layers.size();  // Return the number of layers
}

void Model::set_optimizer(Optimizer& opt) {
    optimizer = &opt;
}

void Model::set_train() {
    training = true;
    for (auto& layer : layers) {
        layer->set_training(true);
    }
}

void Model::set_inference() {
    training = false;
    for (auto& layer : layers) {
        layer->set_training(false);
    }
}

Tensor Model::forward(const Tensor& input) {
   Tensor output = input;
    for (auto& layer : layers) {
        output = layer->forward(output);
    }
    return output;
}

void Model::backward(const Tensor& grad_output) {
    Tensor grad = grad_output;
    for (auto it = layers.rbegin(); it != layers.rend(); ++it) {
        grad = (*it)->backward(grad);
    }
}

void Model::optimize() const {
    for (auto& layer : layers) {
        optimizer->optimize(*layer);
    }
}

void Model::Train(const Tensor& training_data, const Tensor& training_labels, int epochs, Loss& loss_function, Optimizer* optimizer) {
    set_train();
    // Optimizer can either be set in the model or passed as an argument
    if (optimizer != nullptr) {
        set_optimizer(*optimizer);
    } else if (this->optimizer == nullptr) {
        Console::log("No Optimizer provided. Training aborted.", Console::ERROR);
        return;
    }

    // Split the data into batches
    int num_batches = training_data.depth();
    bool stop_training = false;
    Tensor inputs;
    Tensor targets;
    for (int epoch = 0; epoch < epochs; ++epoch) {

        // Handle batches
        float total_loss = 0.0f;
        for (int i = 0; i < num_batches; ++i) {
            // Get the current batch
            inputs = training_data.slice(i);
            targets = training_labels.slice(i);
            // Forward pass
            Tensor outputs = forward(inputs);
            float loss = loss_function.forward(outputs, targets);
            total_loss += loss;

            // Backward pass
            Tensor grad_loss = loss_function.backward();
            backward(grad_loss);
            // Optimize
            optimize();
        }
        // Calculate average loss for the batch
        float average_loss = total_loss / num_batches;

        // Notify callbacks at the end of the epoch
        for (auto& callback : callbacks) {
            callback->on_epoch_end(epoch, average_loss);

            if (auto early_stopping = dynamic_cast<EarlyStopping*>(callback)) {
                if (early_stopping->should_stop()) {
                    stop_training = true;
                }
            }
        }

        if (stop_training) {
            std::cout << "Stopping early at epoch " << epoch << "!" << std::endl;
            break;
        }

        std::cout << "Epoch " << epoch << " completed. Average Loss: " << average_loss << std::endl;
    }
}

// TOOD: This does not really accept true tensors, only matrices
void Model::Test(const Tensor& testing_data, const Tensor& testing_labels, Loss& loss_function) {
    set_inference();
    Tensor outputs = forward(testing_data);
    float loss = loss_function.forward(outputs, testing_labels);
    int correct_predictions = 0;
    for (int i = 0; i < outputs[0].rows(); ++i) {
        // Implement argmax manually
        Eigen::Index predicted_index;
        outputs[0].row(i).maxCoeff(&predicted_index);
        
        Eigen::Index actual_index;
        testing_labels[0].row(i).maxCoeff(&actual_index);
        
        // Compare if the predicted and actual labels match
        if (predicted_index == actual_index) {
            correct_predictions++;
        }
    }
    float accuracy = static_cast<float>(correct_predictions) / outputs[0].rows();
    std::cout << "Test Loss: " << loss << std::endl;
    std::cout << "Test Accuracy: " << accuracy * 100 << "%" << std::endl;
}
