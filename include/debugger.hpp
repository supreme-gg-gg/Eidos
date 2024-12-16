#ifndef DEBUGGER_HPP
#define DEBUGGER_HPP

#include <vector>
#include <Eigen/Dense>
#include "layer.h"
#include <iostream>

class Debugger {
public:
    // Add a new layer to track
    void track_layer(Layer* layer) {
        layers.push_back(layer);
    }

    // Save the current weights of tracked layers
    void save_previous_weights() {
        previous_weights.clear();  // Clear old data
        for (auto layer : layers) {
            previous_weights[layer] = layer->get_weights();  // Save current weights
        }
    }

    // Print the norm of weight changes for each tracked layer
    void print_weight_change_norms() {
        for (auto layer : layers) {
            auto current_weights = layer->get_weights();
            const auto& prev_weights = previous_weights[layer];
            
            double total_norm = 0.0;

            for (size_t i = 0; i < current_weights.size(); ++i) {
                total_norm += (*current_weights[i] - *prev_weights[i]).squaredNorm();
            }

            total_norm = std::sqrt(total_norm);

            std::cout << "Layer: " << layer->get_name() << " | Weight change norm = " << total_norm << std::endl;
        }
    }

private:
    std::vector<Layer*> layers;  // Pointers to tracked layers
    std::unordered_map<Layer*, std::vector<Eigen::MatrixXf*>> previous_weights;  // Previous weights for each layer
};

#endif // DEBUGGER_HPP