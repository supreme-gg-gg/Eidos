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
#include "callback.h"
#include "tensor.hpp"
#include "preprocessors.h"

/**
 * @class Model
 * @brief Represents a neural network model consisting of multiple layers and an optimizer.
 * 
 * The Model class provides functionalities to add layers, set an optimizer, perform forward
 * and backward passes, and optimize the model parameters. It is designed to be used in
 * machine learning tasks where a neural network model is trained and evaluated.
 */
class Model {
private:
    std::vector<std::unique_ptr<Layer> > layers; // Vector of unique pointers to layers
    std::vector<Callback*> callbacks;  // List of callbacks
    Optimizer* optimizer; // Pointer to the optimizer used for training the model
    Loss* loss_function; // Pointer to the loss function used for training the model
    bool training = true;

public:

    /**
     * @brief Constructs a new Model object with default values.
     * 
     * Initializes the optimizer and loss_function pointers to nullptr.
     */
    Model() : optimizer(nullptr), loss_function(nullptr) {};

    /**
     * @brief Constructs a new Model object with the specified optimizer and loss function.
     * 
     * @param optimizer Pointer to the optimizer to be used by the model.
     * @param loss_function Pointer to the loss function to be used by the model.
     * 
     * @note The optimizer and loss function can be set later using the `set_optimizer` and `set_loss_function` methods.
     */
    Model(Optimizer& optimizer, Loss& loss_function) : optimizer(&optimizer), loss_function(&loss_function) {}

    /**
     * @brief Adds a new layer to the model.
     * 
     * @param layer Pointer to the layer to be added.
     */
    void Add(Layer* layer); 

    /**
     * @brief Adds a callback to the list of callbacks.
     * 
     * This function appends the given callback to the internal list of callbacks.
     * 
     * @param callback A pointer to the Callback object to be added.
     */
    void add_callback(Callback* callback);

    /**
     * @brief Sets the optimizer for the model.
     * 
     * @param opt Reference to the optimizer to be used.
     */
    void set_optimizer(Optimizer& opt);

    /**
     * @brief Sets the loss function for the model.
     * 
     * @param loss Reference to the loss function to be used.
     */
    void set_loss_function(Loss& loss);
    
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
    Tensor forward(const Tensor& input);

    /**
     * @brief Performs the backward pass of the model, computing the gradient of the loss with respect to the model's parameters.
     * 
     * This function takes the gradient of the loss with respect to the output of the model and computes the gradient of the loss
     * with respect to the model's parameters. This is typically used in the training process to update the model's parameters.
     * 
     * @param grad_output The gradient of the loss with respect to the output of the model. This is a matrix of the same shape as the model's output.
     */
    void backward(const Tensor& grad_output);

    /**
     * @brief Performs the backward pass of the model, computing the gradient of the loss with respect to the model's parameters.
     * 
     * @note This method requires that a loss function has been set already. Otherwise, you can manualy
     * compute the loss gradient and pass it to the overloaded `backward` method.
     */
    void backward();

    /**
     * @brief Optimizes the model parameters.
     * 
     * This function performs optimization on the model parameters to improve 
     * performance. The specific optimization algorithm and its details are 
     * implementation-dependent.
     */
    void optimize() const;

    /**
     * @brief Sets the model to training mode.
     *
     * This function configures the model to be in training mode, enabling
     * features such as dropout and batch normalization that are typically
     * used during the training process.
     * 
     * @note This function should be called before training the model if
     * you are using manual training loop. Otherwise, the `Train` function
     * will automatically set the model to training mode.
     */
    void set_train();

    /**
     * @brief Sets the model to inference mode.
     *
     * This function configures the model to operate in inference mode,
     * which is typically used for making predictions or classifications
     * based on the trained model.
     * 
     * @note This function should be called before using the model for
     * inference if you are using manual inference loop. Otherwise, the
     * `Test` function will automatically set the model to inference mode.
     */
    void set_inference();

    /**
     * @brief Trains the model using the provided training data and labels.
     * 
     * @warning This does not support batched images (i.e. 4D tensors). 3D tensor inputs are
     * assumed to be a batch of 2D inputs. This will result in unexpected behavior if the
     * input tensor is not a batch of 2D inputs.
     * 
     * @param training_data A matrix containing the training data.
     * @param training_labels A matrix containing the training labels.
     * @param epochs The number of epochs to train the model.
     * @param loss_function The loss function to be used during training.
     * @param optimizer An optional optimizer to be used during training. If not provided, a default optimizer will be used.
     */
    void Train(const Tensor& training_data, const Tensor& training_labels, 
        int epochs, Loss& loss_function, Optimizer* optimizer=nullptr,
        std::vector<Callback*> callbacks=std::vector<Callback*>());

    /**
     * @brief Tests the model using the provided testing data and labels.
     * 
     * @note This assumes that the data is not batched for training. It 
     * treats each Tensor as a single image. There is no batching.
     * 
     * @param data A ImageInputData object containing the training input and labels.
     */
    
    void Train(const ImageInputData& data,
        int epochs, Loss& loss_function, Optimizer* optimizer=nullptr,
        std::vector<Callback*> callbacks=std::vector<Callback*>());

    /**
     * @brief Tests the model using the provided testing data and labels.
     * 
     * @note This assumes that the data is not batched for training.
     * 
     * @param testing_data A tensor containing the testing data.
     * @param testing_labels A tensor containing the testing labels.
     * @param loss_function The loss function to evaluate the model's performance.
     */
    void Test(const Tensor& testing_data, const Tensor& testing_labels, Loss& loss_function);

    /**
     * @brief Tests the model using the provided image input data and computes the loss.
     * 
     * @param data The image input data to be used for testing the model.
     * @param loss_function The loss function to be used for computing the loss.
     */
    void Test(ImageInputData& data, Loss& loss_function);

    /**
     * @brief Retrieves a pointer to the layer at the specified index.
     * 
     * @param index The index of the layer to retrieve.
     * @return Layer* A pointer to the layer at the specified index.
     */
    Layer* get_layer(size_t index) const;

    /**
     * @brief Get the number of layers in the model.
     * 
     * @return The number of layers as a size_t.
     */
    size_t num_layers() const;

    void Serialize(std::string toFilePath);
    void Deserialize(std::string fromFilePath);

    ~Model() = default;
};

#endif // MODEL_H