#ifndef IMAGE_LOADER_H
#define IMAGE_LOADER_H

#include "../include/preprocessors/data_loader.h"
#include <string>
#include <vector>
#include <Eigen/Dense>
#include <map>

/**
 * @class ImageLoader
 * @brief A class for loading and preprocessing image data.
 * 
 * The ImageLoader class is responsible for loading image data from files, preprocessing the data, and splitting it into training and testing sets.
 * It inherits from the DataLoader class, which provides a generic interface for data loading operations.
 * 
 * Example usage:
 * @code
 * #include "image_loader.h"
 * 
 * int main() {
 *     ImageLoader loader;
 *     std::vector<std::vector<Eigen::MatrixXf>> dataSamples;
 *     std::vector<std::string> labels;
 *     
 *     // Load image data from a CSV file
 *     loader.load_data("path/to/your/image_data.csv", dataSamples, labels);
 *     
 *     // Split the data into training and testing sets
 *     std::vector<std::vector<Eigen::MatrixXf>> train_features, test_features;
 *     std::vector<std::string> train_labels, test_labels;
 *     float trainToTestSplitRatio = 0.8f; // 80% training, 20% testing
 *     loader.split_data(features, labels, train_features, train_labels, test_features, test_labels, trainToTestSplitRatio);
 *     
 *     // Now you can use train_features, train_labels, test_features, and test_labels for your machine learning model
 *     
 *     return 0;
 * }
 * @endcode
 * @note This class assumes that the image data is stored in a CSV file with the following format:
 * @code
 * label,image_path
 * class1,path/to/image1.jpg
 * class2,path/to/image2.png
 * class1,path/to/image3.bmp
 * ...
 * @endcode
 * The first column should contain the class labels, and the second column should contain the file paths to the images.
 * @note This class uses netpbm utilities (e.g., jpegtopnm, pngtopnm) for image processing. Make sure these utilities are installed on your system.
 */
class ImageLoader : public DataLoader<std::vector<Eigen::MatrixXf>> {
public:
    /**
     * @brief Loads image data from a specified file.
     * 
     * This function loads image data from the given file path and stores the data in the provided vectors for features and labels.
     * 
     * @param filePath The path to the file from which to load data.
     * @param features A vector to store the loaded feature matrices.
     * @param labels A vector to store the corresponding labels.
     */
    void load_data(const std::string& filePath, 
        std::vector<std::vector<Eigen::MatrixXf>>& features, 
        std::vector<std::string>& labels) override;

    /**
     * @brief Splits the dataset into training and testing sets based on the given ratio.
     * 
     * @param features A vector of vectors containing Eigen::MatrixXf objects representing the features of the dataset.
     * @param labels A vector of strings representing the labels corresponding to the features.
     * @param train_features A reference to a vector of vectors where the training features will be stored.
     * @param train_labels A reference to a vector where the training labels will be stored.
     * @param test_features A reference to a vector of vectors where the testing features will be stored.
     * @param test_labels A reference to a vector where the testing labels will be stored.
     * @param trainToTestSplitRatio A float representing the ratio of the dataset to be used for training. The remaining part will be used for testing.
     */
    void split_data(const std::vector<std::vector<Eigen::MatrixXf>>& features, 
        const std::vector<std::string>& labels, 
        std::vector<std::vector<Eigen::MatrixXf>>& train_features, 
        std::vector<std::string>& train_labels, 
        std::vector<std::vector<Eigen::MatrixXf>>& test_features, 
        std::vector<std::string>& test_labels, 
        float trainToTestSplitRatio) override;

    /**
     * @brief Converts string labels to one-hot encoded labels.
     * 
     * This function takes a vector of string labels and converts them into one-hot encoded labels.
     * The conversion is done using a mapping object of type T2, which should be a key-value object
     * where values are accessible using T2[key] (e.g., map, unordered_map etc.).
     * 
     * @tparam T2 The type of the mapping object used for conversion.
     * @param labels A vector of string labels to be converted.
     * @param one_hot_labels A vector to store the resulting one-hot encoded labels.
     * @param mapping A key-value object used to map string labels to their corresponding one-hot encoded values.
     */
    void convert_to_one_hot(const std::vector<std::string>& labels, 
        std::vector<Eigen::MatrixXf>& one_hot_labels, 
        const std::map<std::string, int>& mapping) override ;
};

#endif // IMAGE_LOADER_H