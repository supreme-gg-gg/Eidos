#ifndef CALLBACK_H
#define CALLBACK_H

#include <iostream>
#include <limits>

class Callback {
public:
    virtual void on_epoch_end(int epoch, float loss) = 0;
};

class EarlyStopping : public Callback {
private:
    float best_loss;
    int patience;
    int epochs_since_improvement;
    bool stop_training;

public:
    EarlyStopping(float patience = 10) 
        : best_loss(std::numeric_limits<float>::infinity()), 
          patience(patience), 
          epochs_since_improvement(0), 
          stop_training(false) {}

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

    bool should_stop() const {
        return stop_training;
    }
};

#endif //CALLBACK_H