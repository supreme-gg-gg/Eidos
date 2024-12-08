#ifndef MODEL_H
#define MODEL_H

#include <string>
#include <map>
#include <variant>
#include <Eigen/Dense>
#include <vector>
#include <memory>
#include "layer.h"
#include "optimizer.h"
#include "loss.h"

const unsigned short IMAGE_CHANNEL_NUM = 3;
const unsigned short IMAGE_XSIZE = 256;
const unsigned short IMAGE_YSIZE = 256;

class ImageSample {
    std::string label;
    Eigen::MatrixXd data[IMAGE_CHANNEL_NUM]; //0: red, 1: green, 2: blue
public:
    ImageSample(std::string label, Eigen::MatrixXd& redMatrix, Eigen::MatrixXd& greenMatrix, Eigen::MatrixXd& blueMatrix);
};

class CNN_Model {
public:
    int loadData(std::string dataLabelsPath, float trainToTestSplitRatio=0.5);
    
    int serializeParameters(std::string toFilePath);
    int deserializeParameters(std::string fromFilePath);
    int Train();
    int Test();
    
    // Prints out some model details for debugging
    std::string Info();
private:
    /*
    Example usage:
    ```
    parameters["C1"] //returns the corresponding parameters matrix or vector for the C1 layer.
    ```
    */
    std::map<std::string, std::variant<Eigen::MatrixXd, Eigen::VectorXd>> parameters;
    
    std::vector<ImageSample> trainingSamples;
    std::vector<ImageSample> testingSamples;
};

/**
 * @class Model
 * @brief Represents a neural network model consisting of multiple layers and an optimizer.
 * 
 * The Model class provides functionalities to add layers, set an optimizer, perform forward
 * and backward passes, and optimize the model parameters. It is designed to be used in
 * machine learning tasks where a neural network model is trained and evaluated.
 * 
 * @note This class uses Eigen library for matrix operations and assumes that the Layer and
 * Optimizer classes are defined elsewhere in the codebase.
 */
class Model {
private:
    std::vector<std::unique_ptr<Layer>> layers; // Vector of unique pointers to layers
    Optimizer* optimizer; // Pointer to the optimizer used for training the model

public:
    /**
     * @brief Adds a new layer to the model.
     * 
     * @param layer Pointer to the layer to be added.
     */
    void Add(Layer* layer); 

    /**
     * @brief Sets the optimizer for the model.
     * 
     * @param opt Reference to the optimizer to be used.
     */
    void set_optimizer(Optimizer& opt);
    
    /**
     * @brief Performs the forward pass of the model.
     * 
     * This function takes an input matrix and computes the forward pass
     * through the model, returning the resulting matrix.
     * 
     * @param input The input matrix of size (n, m) where n is the number of samples
     * and m is the number of features.
     * @return Eigen::MatrixXf The output matrix after the forward pass.
     */
    Eigen::MatrixXf forward(const Eigen::MatrixXf& input);

    /**
     * @brief Performs the backward pass of the model, computing the gradient of the loss with respect to the model's parameters.
     * 
     * This function takes the gradient of the loss with respect to the output of the model and computes the gradient of the loss
     * with respect to the model's parameters. This is typically used in the training process to update the model's parameters.
     * 
     * @param grad_output The gradient of the loss with respect to the output of the model. This is a matrix of the same shape as the model's output.
     */
    void backward(const Eigen::MatrixXf& grad_output);

    /**
     * @brief Optimizes the model parameters.
     * 
     * This function performs optimization on the model parameters to improve 
     * performance. The specific optimization algorithm and its details are 
     * implementation-dependent.
     */
    void optimize() const;

    /**
     * @brief Trains the model using the provided training data and labels.
     * 
     * @param training_data A matrix containing the training data.
     * @param training_labels A matrix containing the training labels.
     * @param epochs The number of epochs to train the model.
     * @param batch_size The size of each batch for training.
     * @param loss_function The loss function to be used during training.
     * @param optimizer An optional optimizer to be used during training. If not provided, a default optimizer will be used.
     */
    void Train(const Eigen::MatrixXf& training_data, const Eigen::MatrixXf& training_labels, int epochs, int batch_size, Loss& loss_function, Optimizer* optimizer=nullptr);

    /**
     * @brief Tests the model using the provided testing data and labels.
     * 
     * @param testing_data A matrix containing the testing data.
     * @param testing_labels A matrix containing the testing labels.
     * @param loss_function The loss function to evaluate the model's performance.
     */
    void Test(const Eigen::MatrixXf& testing_data, const Eigen::MatrixXf& testing_labels, Loss& loss_function);

    void Serialize(std::string toFilePath);
    void Deserialize(std::string fromFilePath);

    ~Model() = default;
};

#endif // MODEL_H