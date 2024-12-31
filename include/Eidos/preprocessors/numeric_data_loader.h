/**
 * @file numeric_data_loader.h
 * @brief Header file for the NumericDataLoader class and related structures.
 */

#ifndef NUMERIC_DATA_LOADER_H
#define NUMERIC_DATA_LOADER_H

#include "data_loader.h"
#include "../tensor.hpp"
#include <string>
#include <vector>
#include <Eigen/Dense>
#include <iterator>
#include <map>
#include <variant>

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
                      const std::string labelsHeaderName, 
                      std::map<std::string, int>& labelsOneHotMapping);

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
    InputData train_test_split(float trainToTestSplitRatio = 0.8f, int batch_size = 1);

    ImageInputData train_test_split_image(size_t reshaped_rows, size_t reshaped_cols, float trainToTestSplitRatio = 0.8f);

    /**
     * @brief Shuffles the data.
     * @return Reference to the current object fluently.
     */
    NumericDataLoader& shuffle();

    NumericDataLoader& linear_transform(float a, float b);

    /**
     * @brief Centers the data around 0.
     * @return Reference to the current object fluently.
     */
    NumericDataLoader& center(float center_val = 0.0);

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
    NumericDataLoader& remove_outliers(float z_threshold = 3.0);

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
     * @brief Gets the number of classes in the labels.
     * @return Number of classes.
     */
    size_t num_classes() const;

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