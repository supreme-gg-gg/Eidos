#ifndef GENERIC_DATA_LOADER_H
#define GENERIC_DATA_LOADER_H

#include "../include/data_loader.h"
#include <string>
#include <vector>
#include <Eigen/Dense>

class GenericDataLoader : public DataLoader<Eigen::MatrixXf> {
public:
    /**
     * @brief Loads data from a specified file.
     * 
     * This function loads data from the given file path and stores the data in the provided vectors for features and labels.
     * 
     * @param filePath The path to the file from which to load data.
     * @param features A vector to store the loaded feature matrices.
     * @param labels A vector to store the corresponding labels.
     */
    void load_data(const std::string& filePath, std::vector<Eigen::MatrixXf>& features, std::vector<std::string>& labels) override ;
    
    /**
     * @brief Splits the dataset into training and testing sets based on the given ratio.
     * 
     * @param features A vector of Eigen::MatrixXf containing the feature matrices.
     * @param labels A vector of strings containing the corresponding labels.
     * @param train_features A vector of Eigen::MatrixXf to store the training feature matrices.
     * @param train_labels A vector of strings to store the training labels.
     * @param test_features A vector of Eigen::MatrixXf to store the testing feature matrices.
     * @param test_labels A vector of strings to store the testing labels.
     * @param trainToTestSplitRatio A float representing the ratio of training data to testing data.
     */
    void split_data(const std::vector<Eigen::MatrixXf>& features, const std::vector<std::string>& labels, std::vector<Eigen::MatrixXf>& train_features, std::vector<std::string>& train_labels, std::vector<Eigen::MatrixXf>& test_features, std::vector<std::string>& test_labels, float trainToTestSplitRatio) override;

    /**
     * @brief Converts string labels to one-hot encoded labels.
     * 
     * This function takes a vector of string labels and converts them into one-hot encoded labels.
     * The conversion is done using a mapping object of type T2, which should be a key-value object
     * where values are accessible using T2[key] (map, unordered_map, etc.).
     * 
     * @tparam T2 The type of the mapping object used for conversion.
     * @param labels A vector of string labels to be converted.
     * @param one_hot_labels A vector to store the resulting one-hot encoded labels.
     * @param mapping A key-value object used to map string labels to their corresponding one-hot encoded values.
     */
    void convert_to_one_hot(const std::vector<std::string>& labels, std::vector<Eigen::MatrixXf>& one_hot_labels, const std::map<std::string, int>& mapping) override ;
};

#endif // GENERIC_DATA_LOADER_H