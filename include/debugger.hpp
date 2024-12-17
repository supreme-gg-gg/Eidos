#ifndef DEBUGGER_HPP
#define DEBUGGER_HPP

#include <vector>
#include <Eigen/Dense>
#include "layer.h"
#include <iostream>
#include <chrono>

/**
 * @class Debugger
 * @brief A class for tracking and debugging layers in a neural network.
 * 
 * The Debugger class provides functionality to monitor the state and behavior
 * of layers during the execution of a neural network program. It allows tracking
 * of layers, saving and comparing weights, and printing norms of weight changes
 * and gradients.
 */
class Debugger {
public:
    /**
     * @brief Tracks the specified layer for debugging purposes.
     * 
     * This function is used to monitor the state and behavior of a given layer
     * during the execution of the program. It can be useful for debugging and
     * ensuring that the layer is functioning as expected.
     * 
     * @param layer A pointer to the Layer object that needs to be tracked.
     */
    void track_layer(Layer* layer) {
        layers.push_back(layer);
    }

    /**
     * @brief Saves the current weights of all layers to the previous_weights map.
     * 
     * This function clears any previously stored weights and then iterates through
     * each layer in the layers vector. For each layer, it retrieves the current weights,
     * creates copies of these weights, and stores them in the previous_weights map
     * with the layer as the key.
     */
    void save_previous_weights() {
        previous_weights.clear();  // Clear old data
        for (auto layer : layers) {
            const auto& weights = layer->get_weights();
            std::vector<Eigen::MatrixXf> weight_copies;
            for (const auto* weight : weights) {
                weight_copies.push_back(*weight);  // Store a copy of the matrix
            }
            previous_weights[layer] = weight_copies;
        }
    }

    /**
     * @brief Prints the norm of the weight changes for each layer.
     *
     * This function iterates through each layer in the neural network, calculates
     * the norm of the difference between the current weights and the previous weights,
     * and prints the result. The norm is calculated as the square root of the sum of
     * squared differences for each weight.
     *
     * The output is in the format:
     * "Layer: <layer_name> | Weight change norm = <norm_value>"
     */
    void print_weight_change_norms() {
        for (auto layer : layers) {
            auto current_weights = layer->get_weights();
            const auto& prev_weights = previous_weights[layer];

            double total_norm = 0.0;

            for (size_t i = 0; i < current_weights.size(); ++i) {
                total_norm += (*current_weights[i] - prev_weights[i]).squaredNorm();
            }

            total_norm = std::sqrt(total_norm);

            std::cout << "Layer: " << layer->get_name() << " | Weight change norm = " << total_norm << std::endl;
        }
    }

    /**
     * @brief Prints the gradient norms for each layer in the neural network.
     *
     * This function iterates through all the layers in the neural network, retrieves
     * the weight gradients for each layer, computes the norm of these gradients, and
     * prints the result. The norm is calculated as the square root of the sum of the
     * squared norms of all gradients.
     */
    void print_gradient_norms() {
        for (auto layer : layers) {
            auto gradients = layer->get_grad_weights();

            double total_norm = 0.0;

            for (const auto& grad : gradients) {
                total_norm += grad->squaredNorm();  // Sum squared norms of all gradients
            }

            total_norm = std::sqrt(total_norm);  // Take square root to compute the norm

            std::cout << "Layer: " << layer->get_name() << " | Gradient norm = " << total_norm << std::endl;
        }
    }

private:
    std::vector<Layer*> layers;  // Pointers to tracked layers
    std::unordered_map<Layer*, std::vector<Eigen::MatrixXf>> previous_weights;  // Copies of previous weights
};

/**
 * @class Timer
 * @brief A simple RAII timer class that measures the duration of its lifetime.
 *
 * The Timer class starts timing upon construction and stops timing upon destruction.
 * The elapsed time is printed to the standard output in milliseconds.
 *
 * Example usage:
 * @code
 * {
 *     Timer t;
 *     // Code to measure goes here
 * }
 * // Timer will automatically stop and print the duration when it goes out of scope.
 * @endcode
 */
class Timer {
public:
    /**
     * @brief Constructs a Timer object and initializes the start time point to the current time.
     */
    Timer() : start_time_point(std::chrono::high_resolution_clock::now()) {}

    /**
     * @brief Destructor for the Timer class.
     * 
     * This destructor calculates the duration between the creation of the Timer object
     * and its destruction. It uses high-resolution clock to get the end time point,
     * then calculates the duration in milliseconds and prints it to the standard output.
     */
    ~Timer() {
        auto end_time_point = std::chrono::high_resolution_clock::now();
        auto start = std::chrono::time_point_cast<std::chrono::milliseconds>(start_time_point).time_since_epoch().count();
        auto end = std::chrono::time_point_cast<std::chrono::milliseconds>(end_time_point).time_since_epoch().count();
        auto duration = end - start;
        std::cout << "Duration: " << duration << " ms" << std::endl;
    }

private:
    std::chrono::time_point<std::chrono::high_resolution_clock> start_time_point;
};

#endif // DEBUGGER_HPP