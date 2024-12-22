#ifndef CALLBACK_H
#define CALLBACK_H

#include <iostream>
#include <limits>

/**
 * @class Callback
 * @brief Abstract base class for defining callback methods in a deep learning framework.
 *
 * This class provides an interface for callback methods that can be used to monitor
 * and respond to events during the training process of a deep learning model.
 */
class Callback {
public:
    virtual void on_epoch_end(int epoch, float loss) = 0;
    virtual bool should_stop() const { return false;};
    virtual ~Callback() {}
};

/**
 * @class EarlyStopping
 * @brief A callback to stop training when a monitored metric has stopped improving.
 *
 * This class implements early stopping to halt training when the loss has not improved
 * for a specified number of epochs (patience).
 * 
 * @param patience The number of epochs with no improvement after which training will be stopped.
 *
 * @details
 * The EarlyStopping class inherits from the Callback class and monitors the loss at the end
 * of each epoch. If the loss does not improve for a number of consecutive epochs specified
 * by the patience parameter, training is stopped.
 */
class EarlyStopping : public Callback {
private:
    float best_loss;
    int patience;
    int epochs_since_improvement;
    bool stop_training;

public:
    /**
     * @brief A callback class for early stopping during training.
     * 
     * This class monitors the loss during training and stops the training process
     * if the loss does not improve for a specified number of epochs (patience).
     * 
     * @param patience The number of epochs to wait for an improvement in loss before stopping training. Default is 10.
     */
    EarlyStopping(float patience = 10) 
        : best_loss(std::numeric_limits<float>::infinity()), 
          patience(patience), 
          epochs_since_improvement(0), 
          stop_training(false) {}

    /**
     * @brief Callback function called at the end of each epoch during training.
     *
     * This function is used to monitor the training process and implement early stopping.
     * It updates the best loss observed so far and counts the number of epochs since the last improvement.
     * If the number of epochs without improvement exceeds the patience threshold, training is stopped.
     *
     * @param epoch The current epoch number.
     * @param loss The loss value at the end of the current epoch.
     */
    void on_epoch_end(int epoch, float loss) override {
        if (loss < best_loss) {
            best_loss = loss;  // Update best loss
            epochs_since_improvement = 0;  // Reset counter
        } else {
            epochs_since_improvement++;  // Increment if no improvement
        }

        if (epochs_since_improvement >= patience) {
            stop_training = true;
            std::cout << "Early stopping triggered at epoch " << epoch << std::endl;
        }
    }

    /**
     * @brief Checks if the training process should be stopped.
     * 
     * @return true if the training process should be stopped, false otherwise.
     */
    bool should_stop() const {
        return stop_training;
    }
};

/**
 * @class PrintLoss
 * @brief A callback class for printing the loss at specified intervals during training.
 * 
 * This class inherits from the Callback base class and provides functionality to print
 * the loss value at the end of an epoch if the epoch number is a multiple of the specified
 * print interval.
 */
class PrintLoss : public Callback {
private:
    int print_interval;
public:
    /**
     * @brief Constructor for the PrintLoss class.
     * 
     * @param print_interval The interval at which the loss should be printed.
     */
    PrintLoss(int print_interval) : print_interval(print_interval) {}

    /**
     * @brief Callback function called at the end of each epoch.
     *
     * This function is called at the end of each epoch during training. It prints
     * the epoch number and the loss value if the current epoch is a multiple of
     * the print interval.
     *
     * @param epoch The current epoch number.
     * @param loss The loss value at the end of the current epoch.
     */
    void on_epoch_end(int epoch, float loss) override {
        if (epoch % print_interval == 0) {
            std::cout << "Epoch: " << epoch << " Loss: " << loss << std::endl;
        }
    }
};

#endif //CALLBACK_H