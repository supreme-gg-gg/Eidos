#include "../include/model.h"
#include "../include/console.hpp"
#include "../include/csvparser.h"
#include "../include/layer.h"
#include "../include/activation_fns.h"
#include "../include/optimizer.h"
#include <fstream>
#include <filesystem>
#include <string>
#include <vector>
#include <algorithm>
#include <random>
#include <Eigen/Dense>

void Model::Add(Layer* layer) {
    layers.emplace_back(layer); // Wraps raw pointer in a unique_ptr
}

void Model::add_callback(Callback* callback) {
    callbacks.push_back(callback);
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

Eigen::MatrixXf Model::forward(const Eigen::MatrixXf& input) {
    Eigen::MatrixXf output = input;
    for (auto& layer : layers) {
        output = layer->forward(output);
    }
    return output;
}

void Model::backward(const Eigen::MatrixXf& grad_output) {
    Eigen::MatrixXf grad = grad_output;
    for (auto it = layers.rbegin(); it != layers.rend(); ++it) {
        grad = (*it)->backward(grad);
    }
}

void Model::optimize() const {
    for (auto& layer : layers) {
        optimizer->optimize(*layer);
    }
}

void Model::Train(const Eigen::MatrixXf& training_data, const Eigen::MatrixXf& training_labels, int epochs, int batch_size, Loss& loss_function, Optimizer* optimizer) {
    set_train();
    // Optimizer can either be set in the model or passed as an argument
    if (optimizer != nullptr) {
        set_optimizer(*optimizer);
    } else if (this->optimizer == nullptr) {
        Console::log("No Optimizer provided. Training aborted.", Console::ERROR);
        return;
    }

    // Split the data into batches
    int num_batches = training_data.rows() / batch_size;
    bool stop_training = false;
    Eigen::MatrixXf inputs;
    Eigen::MatrixXf targets;
    for (int epoch = 0; epoch < epochs; ++epoch) {

        // Handle batches
        float total_loss = 0.0f;
        for (int i = 0; i < num_batches; ++i) {
            // Get the current batch
            inputs = training_data.middleRows(i * batch_size, batch_size);
            targets = training_labels.middleRows(i * batch_size, batch_size);
            // Forward pass
            Eigen::MatrixXf outputs = forward(inputs);
            float loss = loss_function.forward(outputs, targets);
            total_loss += loss;

            // Backward pass
            Eigen::MatrixXf grad_loss = loss_function.backward();
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

void Model::Test(const Eigen::MatrixXf& testing_data, const Eigen::MatrixXf& testing_labels, Loss& loss_function) {
    set_inference();
    Eigen::MatrixXf outputs = forward(testing_data);
    float loss = loss_function.forward(outputs, testing_labels);
    Softmax softmax;
    outputs = softmax.forward(outputs);
    int correct_predictions = 0;
    for (int i = 0; i < outputs.rows(); ++i) {
        // Implement argmax manually
        Eigen::Index predicted_index;
        outputs.row(i).maxCoeff(&predicted_index);
        
        Eigen::Index actual_index;
        testing_labels.row(i).maxCoeff(&actual_index);
        
        // Compare if the predicted and actual labels match
        if (predicted_index == actual_index) {
            correct_predictions++;
        }
    }
    float accuracy = static_cast<float>(correct_predictions) / outputs.rows();
    std::cout << "Test Loss: " << loss << std::endl;
    std::cout << "Test Accuracy: " << accuracy * 100 << "%" << std::endl;
}
