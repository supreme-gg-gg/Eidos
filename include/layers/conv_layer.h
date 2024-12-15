// #ifndef CONVOLUTIONAL_LAYER_H
// #define CONVOLUTIONAL_LAYER_H

// #include <Eigen/Dense>
// #include <vector>
// #include "layer.h"

// class ConvolutionalLayer : public Layer {
// private:
//     std::vector<Eigen::MatrixXf> weights;    // Weights for each filter
//     std::vector<Eigen::VectorXf> biases;     // Biases for each output channel
//     std::vector<Eigen::MatrixXf> grad_weights; // Gradients of weights
//     std::vector<Eigen::VectorXf> grad_biases;  // Gradients of biases
//     std::vector<Eigen::MatrixXf> input;      // Stored input for backpropagation

//     int input_channels_;  // Number of input channels (depth of input feature maps)
//     int output_channels_; // Number of output channels (depth of output feature maps)
//     int kernel_size_;     // Size of the convolutional kernel
//     int stride_;          // Stride for the convolution
//     int padding_;         // Padding for the convolution

// public:
//     // Constructor
//     ConvolutionalLayer(int input_channels, int output_channels, 
//                         int kernel_size, int stride = 1, int padding = 0);

//     // Forward pass
//     std::vector<Eigen::MatrixXf> forward(const std::vector<Eigen::MatrixXf>& input) override {
//         return forward(input[0]);
//     };

//     // Backward pass (gradient computation)
//     std::vector<Eigen::MatrixXf> backward(const std::vector<Eigen::MatrixXf>& grad_output) override {
//         return backward(grad_output[0]);
//     };

//     // Getter methods
//     bool has_weights() const override { return true; };
//     bool has_bias() const override { return true; };

//     std::vector<Eigen::MatrixXf>* get_weights() override {
//         return &weights;
//     };
//     std::vector<Eigen::VectorXf>* get_biases() override {
//         return &biases;
//     };
//     std::vector<Eigen::MatrixXf>* get_grad_weights() override {
//         return &grad_weights;
//     };
//     std::vector<Eigen::VectorXf>* get_grad_biases() override {
//         return &grad_biases;
//     };
// };

// #endif // CONVOLUTIONAL_LAYER_H