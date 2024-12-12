#include "../include/model.h"
#include "../include/console.hpp"
#include "../include/csvparser.h"
#include "../include/layer.h"
#include "../include/dense_layer.h"
#include "../include/activation_fns.h"
#include "../include/optimizer.h"
#include "../include/tensor.h"
#include <string>
#include <vector>
#include <algorithm>
#include <random>
#include <Eigen/Dense>

template <typename LayerType>
void Model::Add(LayerType* layer) {
    layers.emplace_back(layer); // Wraps raw pointer in a unique_ptr
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

template <typename T = Eigen::MatrixXf>
T Model::forward(const T& input) {
    T output = input;
    for (auto& layer : layers) {
        output = layer->forward(output);
    }
    return output;
}

template <typename T = Eigen::MatrixXf>
void Model::backward(const T& grad_output) {
    T grad = grad_output;
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
    Eigen::MatrixXf inputs;
    Eigen::MatrixXf targets;
    for (int epoch = 0; epoch < epochs; ++epoch) {
        for (int i = 0; i < num_batches; ++i) {
            // Get the current batch
            inputs = training_data.middleRows(i * batch_size, batch_size);
            targets = training_labels.middleRows(i * batch_size, batch_size);
            // Forward pass
            Eigen::MatrixXf outputs = forward(inputs);
            float loss = loss_function.forward(outputs, targets);
            // Backward pass
            Eigen::MatrixXf grad_loss = loss_function.backward();
            backward(grad_loss);
            // Optimize
            optimize();
            // Print loss
            std::cout << "Epoch " << epoch << " Batch " << i << " completed. Loss: " << loss << std::endl;
        }
    }
}

void Model::Train(const Tensor<Eigen::MatrixXf>& training_data, const Tensor<Eigen::MatrixXf>& training_labels, int epochs, int batch_size, Loss& loss_function, Optimizer* optimizer) {
    return;
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

void Model::Test(const Tensor<Eigen::MatrixXf>& testing_data, const Tensor<Eigen::MatrixXf>& testing_labels, Loss& loss_function) {
    return;
}