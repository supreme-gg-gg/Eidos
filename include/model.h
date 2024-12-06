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

class Model {
private:
    std::vector<std::unique_ptr<Layer>> layers;

public:
    void add_dense_layer(int input_size, int output_size);
    void add_relu_layer();
    
    Eigen::MatrixXf forward(const Eigen::MatrixXf& input);
    void backward(const Eigen::MatrixXf& grad_output);
    void optimize(Optimizer& optimizer);
};

#endif // MODEL_H