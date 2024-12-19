/**
 * @file numeric_data_loader.h
 * @brief Header file for the NumericDataLoader class and related structures.
 */

#ifndef NUMERIC_DATA_LOADER_H
#define NUMERIC_DATA_LOADER_H

#include <string>
#include <vector>
#include <Eigen/Dense>
#include <iterator>
#include <map>

/**
 * @struct InputData
 * @brief Structure to hold training and testing data.
 */
struct InputData {
    Eigen::MatrixXf train_features; ///< Training features matrix
    Eigen::MatrixXf train_labels;   ///< Training labels matrix
    Eigen::MatrixXf test_features;  ///< Testing features matrix
    Eigen::MatrixXf test_labels;    ///< Testing labels matrix
};

/**
 * @struct BatchInputData
 * @brief Placeholder structure for batch input data.
 */
struct BatchInputData {};

/**
 * @class NumericDataLoader
 * @brief Class for loading and preprocessing numeric data.
 */
class NumericDataLoader {
public:
    /**
     * @brief Constructor to load data from a CSV file.
     * @param filePath Path to the CSV file.
     * @param labelsHeaderName Header name for labels in the file.
     * @param shuffle Whether to shuffle the data.
     */
    NumericDataLoader(const std::string& filePath, 
                      const std::string labelsHeaderName = "labels", 
                      bool shuffle = true);

    /**
     * @brief Constructor to initialize with features and labels.
     * @param features Matrix of features.
     * @param labels Matrix of labels.
     */
    NumericDataLoader(const Eigen::MatrixXf& features, const Eigen::MatrixXf& labels);

    /**
     * @brief Splits the data into training and testing sets.
     * @param trainToTestSplitRatio Ratio of training to testing data.
     * @return InputData structure containing split data.
     */
    InputData train_test_split(float trainToTestSplitRatio = 0.8f);

    /**
     * @brief Shuffles the data.
     * @return Reference to the current object fluently.
     */
    NumericDataLoader& shuffle();

    /**
     * @brief Scales the data by a coefficient.
     * @param coefficient Scaling coefficient.
     * @return Reference to the current object fluently.
     */
    NumericDataLoader& scale(float coefficient = 1.0f);

    /**
     * @brief Centers the data around 0.
     * @return Reference to the current object fluently.
     */
    NumericDataLoader& center();

    /**
     * @brief Applies min-max scaling to the data.
     * @param min_val Minimum value for scaling.
     * @param max_val Maximum value for scaling.
     * @return Reference to the current object fluently.
     */
    NumericDataLoader& min_max_scale(int min_val = 0, int max_val = 1);

    /**
     * @brief Normalizes the data using z-score normalization.
     * @return Reference to the current object fluently.
     */
    NumericDataLoader& z_score_normalize();

    /**
     * @brief Removes outliers from the data.
     * @param z_threshold Z-score threshold for outlier removal.
     * @return Reference to the current object fluently.
     */
    NumericDataLoader& removeOutliers(float z_threshold = 3.0);

    /**
     * @brief Applies Principal Component Analysis (PCA) to the data.
     * @param target_dim Target dimensionality after PCA.
     * @return Reference to the current object fluently.
     */
    NumericDataLoader& pca(int target_dim = 5);

    /**
     * @brief Gets the number of samples in the data.
     * @return Number of samples.
     */
    size_t num_samples() const;

    /**
     * @brief Gets the number of features in the data.
     * @return Number of features.
     */
    size_t num_features() const;

    /**
     * @brief Gets the number of categories in the labels.
     * @return Number of categories.
     */
    size_t num_categories() const;

    /**
     * @brief Gets the shape of the data.
     * @return Pair of integers representing the shape (rows, columns).
     */
    std::pair<int, int> shape() const;

    /**
     * @brief Prints a preview of the data.
     * @param num_columns Number of columns to preview.
     */
    void print_preview(int num_columns = 5) const;

private:
    Eigen::MatrixXf features_; ///< Matrix of features
    Eigen::MatrixXf labels_;   ///< Matrix of labels
    std::map<std::string, int> oneHotMapping_; ///< Mapping for one-hot encoding
};

#endif // NUMERIC_DATA_LOADER_H