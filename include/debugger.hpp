#ifndef DEBUGGER_HPP
#define DEBUGGER_HPP

#include <vector>
#include <Eigen/Dense>
#include "layer.h"
#include <iostream>

class Debugger {
    std::vector<std::vector<Eigen::MatrixXf>> previous_weights;

public:
    // Save current weights of each layer
    void save_weights(const std::vector<Layer*>& layers) {
        previous_weights.clear();  // Clear previous weights before saving

        for (auto* layer : layers) {
            std::vector<Eigen::MatrixXf> layer_weights = layer->get_weights();
            previous_weights.push_back(layer_weights);  // Store current layer weights
        }
    }

    // Retrieve previous weights for a specific layer
    std::vector<Eigen::MatrixXf> get_previous_weights(int layer_index) {
        if (layer_index < 0 || layer_index >= previous_weights.size()) {
            std::cerr << "Invalid layer index" << std::endl;
            return {};  // Return empty vector if the index is invalid
        }
        return previous_weights[layer_index];
    }

    // Compute weight change norm for a specific layer
    float compute_weight_change_norm(Layer* layer, int layer_index) {
        auto current_weights = layer->get_weights();
        auto previous_weights = get_previous_weights(layer_index);

        if (current_weights.size() != previous_weights.size()) {
            std::cerr << "Mismatch between current and previous weights size" << std::endl;
            return 0.0f;
        }

        float norm = 0.0f;
        for (size_t i = 0; i < current_weights.size(); ++i) {
            norm += (current_weights[i]->array() - previous_weights[i].array()).square().sum();
        }
        return std::sqrt(norm);  // L2 norm
    }
};

#endif // DEBUGGER_HPP