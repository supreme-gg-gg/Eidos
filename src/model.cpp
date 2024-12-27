#include "../include/model.h"
#include "../include/console.hpp"
#include "../include/csvparser.h"
#include "../include/layer.h"
#include "../include/layers.h"
#include "../include/activation_fns.h"
#include "../include/optimizer.h"
#include "../include/loss_fns.h"
#include "../include/layers/flatten_layer.h"
#include "../include/tensor.hpp"
#include "../include/preprocessors.h"
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

void Model::set_loss_function(Loss& loss) {
    loss_function = &loss;
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

void Model::backward() {
    if (loss_function == nullptr) {
        Console::log("No loss function provided. Backward pass aborted.", Console::ERROR);
        return;
    }
    backward(loss_function->backward());
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

void Model::Train(const ImageInputData& data,
    int epochs, Loss* loss_function, Optimizer* optimizer,
    std::vector<Callback*> callbacks) {

    if (optimizer != nullptr) {
        set_optimizer(*optimizer);
    } else if (this->optimizer == nullptr) {
        Console::log("No Optimizer provided. Training aborted.", Console::ERROR);
        return;
    }
    if (loss_function != nullptr) {
        set_loss_function(*loss_function);
    } else if (this->loss_function == nullptr) {
        Console::log("No Loss function provided. Training aborted.", Console::ERROR);
        return;
    }
    if (!callbacks.empty()) {
        this->callbacks = callbacks;
    }
    
    set_train();
    bool stop_training = false;
    for (int epoch = 0; epoch < epochs; ++epoch) {
        float total_loss = 0.0;
        for (size_t i = 0; i < data.training.inputs.size(); ++i) {
            Tensor output = forward(data.training.inputs[i]);
            float loss = this->loss_function->forward(output, data.training.targets[i]);
            total_loss += loss;
            backward();
            optimize();
        }
        float average_loss = total_loss / data.training.inputs.size();
        
        // Notify callbacks at the end of the epoch
        for (auto& callback : this->callbacks) {
            callback->on_epoch_end(epoch, average_loss);
            if (callback->should_stop()) {
                stop_training = true;
            }
        }

        if (stop_training) {
            std::cout << "Stopping at epoch " << epoch << "!" << std::endl;
            break;
        }
    }
}

void Model::Train(const Tensor& training_data, const Tensor& training_labels, 
    int epochs, Loss* loss_function, Optimizer* optimizer,
    std::vector<Callback*> callbacks) {

    set_train();
    // Optimizer can either be set in the model or passed as an argument
    if (optimizer != nullptr) {
        set_optimizer(*optimizer);
    } else if (this->optimizer == nullptr) {
        Console::log("No Optimizer provided. Training aborted.", Console::ERROR);
        return;
    }
    // Loss function can either be set in the model or passed as an argument
    if (loss_function != nullptr) {
        set_loss_function(*loss_function);
    } else if (this->loss_function == nullptr) {
        Console::log("No Loss function provided. Training aborted.", Console::ERROR);
        return;
    }
    // Callbacks can be set in the model or passed as an argument
    if (!callbacks.empty()) {
        this->callbacks = callbacks;
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
            float loss = this->loss_function->forward(outputs, targets);
            total_loss += loss;

            // Backward pass
            Tensor grad_loss = this->loss_function->backward();
            backward(grad_loss);
            // Optimize
            optimize();
        }
        // Calculate average loss for the batch
        float average_loss = total_loss / num_batches;

        // Notify callbacks at the end of the epoch
        for (auto& callback : this->callbacks) {
            callback->on_epoch_end(epoch, average_loss);
            if (callback->should_stop()) {
                stop_training = true;
            }
        }

        if (stop_training) {
            std::cout << "Stopping at epoch " << epoch << "!" << std::endl;
            break;
        }
    }
}

void Model::Test(const Tensor& testing_data, const Tensor& testing_labels, Loss* loss_function) {
    if (loss_function != nullptr) {
        set_loss_function(*loss_function);
    } else if (this->loss_function == nullptr) {
        Console::log("No Loss function provided. Testing aborted.", Console::ERROR);
        return;
    }
    
    set_inference();
    Tensor flattened_inputs(testing_data.flatten());
    Tensor flattened_labels(testing_labels.flatten());
    Tensor outputs = forward(flattened_inputs);
    float loss = this->loss_function->forward(outputs, flattened_labels);
    int correct_predictions = 0;
    for (int i = 0; i < outputs.getSingleMatrix().rows(); ++i) {
        // Implement argmax manually
        Eigen::Index predicted_index;
        outputs.getSingleMatrix().row(i).maxCoeff(&predicted_index);
        
        Eigen::Index actual_index;
        flattened_labels.getSingleMatrix().row(i).maxCoeff(&actual_index);
        
        // Compare if the predicted and actual labels match
        if (predicted_index == actual_index) {
            correct_predictions++;
        }
    }
    float accuracy = static_cast<float>(correct_predictions) / outputs[0].rows();
    std::cout << "Test Loss: " << loss << std::endl;
    std::cout << "Test Accuracy: " << accuracy * 100 << "%" << std::endl;
}

void Model::Test(ImageInputData& data, Loss* loss_function) {
    if (loss_function != nullptr) {
        set_loss_function(*loss_function);
    } else if (this->loss_function == nullptr) {
        Console::log("No Loss function provided. Testing aborted.", Console::ERROR);
        return;
    }
    
    set_inference();
    float test_loss = 0.0;
    int correct_predictions = 0;
    for (size_t i = 0; i < data.testing.inputs.size(); ++i) {
        Tensor output = forward(data.testing.inputs[i]);
        float loss = this->loss_function->forward(output, data.testing.targets[i]);
        test_loss += loss;

        // Flatten the output to a vector
        Eigen::VectorXf flat_output = Eigen::Map<Eigen::VectorXf>(output[0].data(), output[0].size());
        int predicted_class = 0;
        flat_output.maxCoeff(&predicted_class); // Get the maximum score index

        // Flatten the target to a vector
        Eigen::VectorXf target_vector = Eigen::Map<Eigen::VectorXf>(data.testing.targets[i][0].data(), data.testing.targets[i][0].size());
        int true_class = 0;
        target_vector.maxCoeff(&true_class); // Get the true class index

        if (predicted_class == true_class) {
            ++correct_predictions;
        }
    }
    float accuracy = static_cast<float>(correct_predictions) / data.testing.inputs.size();
    std::cout << "Test Loss: " << test_loss / data.testing.inputs.size() << std::endl;
    std::cout << "Test Accuracy: " << accuracy * 100.0 << "%" << std::endl;
}

void Model::Serialize(std::string toFilePath, bool override_warning, 
    bool weights_only, bool save_architecture) {
    // Check if the file already exists
    if (std::filesystem::exists(toFilePath)) {
        if (override_warning) {
            Console::log("File already exists.", Console::WARNING);
            std::cout << "Do you want to overwrite the file? (y/n): ";
            char choice;
            std::cin >> choice;
            if (choice != 'y') {
                Console::log("Serialization aborted.", Console::WARNING);
                return;
            }
        }
        // Delete the file if the user chooses to overwrite in order to avoid unexpected behavior
        std::filesystem::remove(toFilePath);
    }
    std::ofstream file(toFilePath, std::ios::binary);
    if (!file.is_open()) {
        Console::log("Failed to open file for serialization", Console::ERROR);
        return;
    }
    
    // Maximum size of layer name
    size_t NameBuffSize = 32;
    file.write(reinterpret_cast<char*>(&NameBuffSize), sizeof(size_t));
    
    size_t num_layers = layers.size();
    // Write the number of layers
    file.write(reinterpret_cast<char*>(&num_layers), sizeof(size_t));
    
    for (auto& layer : layers) {
        // Create a fixed-size buffer of 32 bytes
        char* buffer = new char[NameBuffSize];
        // Copy the layer name into the buffer, truncating if necessary
        std::strncpy(buffer, layer->get_name().c_str(), NameBuffSize - 1);
        // Write the fixed-size buffer to the file
        file.write(buffer, NameBuffSize);
        delete[] buffer;
        layer->serialize(file);
    }

    if (save_architecture) {

        // Remove ".bin" from the file path
        size_t pos = toFilePath.rfind(".bin");
        if (pos != std::string::npos && pos == toFilePath.length() - 4) {
            toFilePath.erase(pos); // Remove ".bin"
        }

        // Write the architecture to a text file
        std::ofstream architectureFile(toFilePath + ".txt", std::ios::out);
        if (!architectureFile.is_open()) {
            Console::log("Failed to open architecture file for writing", Console::ERROR);
            return;
        }
        
        // Write layers, layer info, loss function, and optimizer
        file << "Model Architecture:\n\n# Layers\n" << std::endl;
        for (size_t i = 0; i < num_layers; ++i) {
            architectureFile << "Layer " << i + 1<< ":\n";
            architectureFile << "   Type: " << layers[i]->get_name() << std::endl;
            architectureFile << layers[i]->get_details() << std::endl;
        }

        architectureFile << "# Loss Function:\n";
        if (loss_function != nullptr) {
            architectureFile << "   " + loss_function->get_name() + "\n";
        } else {
            architectureFile << "None" << std::endl;
        }

        architectureFile << "# Optimizer:\n";
        if (optimizer != nullptr) {
            architectureFile << "   " + optimizer->get_name() + "\n";
        } else {
            architectureFile << "None" << std::endl;
        }

        architectureFile.close();
    }

    if (weights_only) {
        file.close();
        return;
    }

    // Write the set loss function and optimizer
    if (loss_function != nullptr) {
        char* buffer = new char[NameBuffSize];
        // Copy the layer name into the buffer, truncating if necessary
        std::strncpy(buffer, loss_function->get_name().c_str(), NameBuffSize - 1);
        // Write the fixed-size buffer to the file
        file.write(buffer, NameBuffSize);
        delete[] buffer;
    } else {
        std::string empty = "";
        char* buffer = new char[NameBuffSize];
        // Copy the layer name into the buffer, truncating if necessary
        std::strncpy(buffer, empty.c_str(), NameBuffSize - 1);
        // Write the fixed-size buffer to the file
        file.write(buffer, NameBuffSize);
        delete[] buffer;
    }

    if (optimizer != nullptr) {
        char* buffer = new char[NameBuffSize];
        // Copy the layer name into the buffer, truncating if necessary
        std::strncpy(buffer, optimizer->get_name().c_str(), NameBuffSize - 1);
        // Write the fixed-size buffer to the file
        file.write(buffer, NameBuffSize);
        delete[] buffer;
        optimizer->serialize(file);
    } else {
        std::string empty = "";
        char* buffer = new char[NameBuffSize];
        // Copy the layer name into the buffer, truncating if necessary
        std::strncpy(buffer, empty.c_str(), NameBuffSize - 1);
        // Write the fixed-size buffer to the file
        file.write(buffer, NameBuffSize);
        delete[] buffer;
    }

    size_t num_callbacks = callbacks.size();
    file.write(reinterpret_cast<char*>(&num_callbacks), sizeof(size_t));
    for (auto& callback : callbacks) {
        char* buffer = new char[NameBuffSize];
        // Copy the layer name into the buffer, truncating if necessary
        std::strncpy(buffer, callback->get_name().c_str(), NameBuffSize - 1);
        // Write the fixed-size buffer to the file
        file.write(buffer, NameBuffSize);
        delete[] buffer;
        callback->serialize(file);
    }

    file.close();
}

void Model::Deserialize(std::string fromFilePath, bool weights_only) {
    std::ifstream file(fromFilePath, std::ios::binary);
    if (!file.is_open()) {
        Console::log("Failed to open file for deserialization", Console::ERROR);
        return;
    }

    size_t NameBuffSize;
    file.read(reinterpret_cast<char*>(&NameBuffSize), sizeof(size_t));
    
    // Clear the current model
    layers.clear();
    callbacks.clear();
    if (!weights_only) {
        loss_function = nullptr;
        optimizer = nullptr;
    }
    
    size_t num_layers;
    file.read(reinterpret_cast<char*>(&num_layers), sizeof(size_t));
    
    for (size_t i = 0; i < num_layers; ++i) {
        char* name = (char*)malloc(NameBuffSize);
        file.read(name, NameBuffSize);
        std::string layer_name(name);
        free(name);
        
        if (layer_name == "Dense") {
            layers.emplace_back(DenseLayer::deserialize(file));
        } else if (layer_name == "Conv2D") {
            layers.emplace_back(Conv2D::deserialize(file));
        } else if (layer_name == "MaxPooling2D") {
            layers.emplace_back(MaxPooling2D::deserialize(file));
        } else if (layer_name == "AveragePooling2D") {
            layers.emplace_back(AveragePooling2D::deserialize(file));
        } else if (layer_name == "RNN") {
            layers.emplace_back(RNNLayer::deserialize(file));
        } else if (layer_name == "GRU") {
            layers.emplace_back(GRULayer::deserialize(file));
        } else if (layer_name == "Flatten") {
            layers.emplace_back(FlattenLayer::deserialize(file));
        } else if (layer_name == "ReLU") {
            layers.emplace_back(ReLU::deserialize(file));
        } else if (layer_name == "LeakyReLU") {
            layers.emplace_back(LeakyReLU::deserialize(file));
        } else if (layer_name == "Sigmoid") {
            layers.emplace_back(Sigmoid::deserialize(file));
        } else if (layer_name == "Softmax") {
            layers.emplace_back(Softmax::deserialize(file));
        } else if (layer_name == "Tanh") {
            layers.emplace_back(Tanh::deserialize(file));
        } else if (layer_name == "Dropout") {
            layers.emplace_back(Dropout::deserialize(file));
        } else if (layer_name == "BatchNorm") {
            layers.emplace_back(BatchNorm::deserialize(file));
        }
        else {
            Console::log("Unknown layer type: " + layer_name, Console::ERROR);
            return;
        }
        Console::log("Layer deserialized: " + layer_name, Console::DEBUG);
    }

    if (weights_only) {
        file.close();
        return;
    }
    
    char* loss_name = (char*)malloc(NameBuffSize);
    file.read(loss_name, NameBuffSize);
    std::string loss_fn_name(loss_name);
    free(loss_name);
    if (loss_fn_name == "") {
        loss_function = nullptr;
    } else if (loss_fn_name == "MSELoss") {
        loss_function = new MSELoss();
    } else if (loss_fn_name == "CrossEntropyLoss") {
        loss_function = new CrossEntropyLoss();
    } else if (loss_fn_name == "CategoricalCrossEntropyLoss") {
        loss_function = new CategoricalCrossEntropyLoss();
    } else if (loss_fn_name == "BinaryCrossEntropyLoss") {
        loss_function = new BinaryCrossEntropyLoss();
    }
    else {
        Console::log("Unknown loss function: " + loss_fn_name, Console::ERROR);
        return;
    }
    
    char* opt_name = (char*)malloc(NameBuffSize);
    file.read(opt_name, NameBuffSize);
    std::string opt_fn_name(opt_name);
    free(opt_name);
    if (opt_fn_name == "") {
        optimizer = nullptr;
    } else if (opt_fn_name == "SGD") {
        optimizer = SGD::deserialize(file);
    } else if (opt_fn_name == "Adam") {
        optimizer = Adam::deserialize(file);
    }
    else {
        Console::log("Unknown optimizer: " + opt_fn_name, Console::ERROR);
        return;
    }

    size_t num_callbacks;
    file.read(reinterpret_cast<char*>(&num_callbacks), sizeof(size_t));
    for (size_t i = 0; i < num_callbacks; ++i) {
        char* callback_name = (char*)malloc(NameBuffSize);
        file.read(callback_name, NameBuffSize);
        std::string callback_fn_name(callback_name);
        free(callback_name);
        if (callback_fn_name == "PrintLoss") {
            callbacks.push_back(PrintLoss::deserialize(file));
        } else if (callback_fn_name == "EarlyStopping") {
            callbacks.push_back(EarlyStopping::deserialize(file));
        } else if (callback_fn_name == "SaveModel") {
            callbacks.push_back(SaveModel::deserialize(file, *this));
        }
        else {
            Console::log("Unknown callback: " + callback_fn_name, Console::ERROR);
            return;
        }
    }

    file.close();
}