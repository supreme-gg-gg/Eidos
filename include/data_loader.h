#ifndef DATA_LOADER_H
#define DATA_LOADER_H

#include <Eigen/Dense>
#include <string>
#include <vector>

class DataLoader {
public:
    virtual ~DataLoader() = default;

    /**
     * @brief Pure virtual function to load data from a specified file.
     * 
     * This function must be implemented by derived classes to load data from the given file path.
     * The loaded data should be stored in the provided vectors for features and labels.
     * 
     * @param filePath The path to the file from which to load data.
     * @param features A vector to store the loaded feature matrices.
     * @param labels A vector to store the corresponding labels.
     */
    virtual void load_data(const std::string& filePath, std::vector<Eigen::MatrixXf>& features, std::vector<std::string>& labels) = 0;
    
    /**
     * @brief Pure virtual function to preprocess data.
     * 
     * This function is responsible for preprocessing the input data features and labels.
     * Derived classes must implement this function to define specific preprocessing steps.
     * 
     * @param features A vector of Eigen::MatrixXf representing the input features to be preprocessed.
     * @param labels A vector of strings representing the labels associated with the input features.
     */
    virtual void preprocess_data(std::vector<Eigen::MatrixXf>& features, std::vector<std::string>& labels) = 0;
    
    /**
     * @brief Splits the dataset into training and testing sets.
     * 
     * @param features A vector of Eigen::MatrixXf containing the feature matrices.
     * @param labels A vector of strings containing the corresponding labels.
     * @param train_features A vector of Eigen::MatrixXf to store the training feature matrices.
     * @param train_labels A vector of strings to store the training labels.
     * @param test_features A vector of Eigen::MatrixXf to store the testing feature matrices.
     * @param test_labels A vector of strings to store the testing labels.
     */
    virtual void split_data(const std::vector<Eigen::MatrixXf>& features, const std::vector<std::string>& labels, std::vector<Eigen::MatrixXf>& train_features, std::vector<std::string>& train_labels, std::vector<Eigen::MatrixXf>& test_features, std::vector<std::string>& test_labels) = 0;
};

#endif // DATA_LOADER_H